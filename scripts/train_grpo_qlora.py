import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import bitsandbytes
import datasets
import peft
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")
FINAL_RE = re.compile(
    r"(?:final\s+answer|최종\s*정답)\s*(?:is|:|=)?\s*"
    r"(?:\\boxed\s*\{\s*)?([+-]?\d[\d,]*)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*([+-]?\d[\d,]*)\s*\}")
ANSWER_RE = re.compile(
    r"(?:answer|정답)\s*(?:is|:|=)?\s*([+-]?\d[\d,]*)",
    re.IGNORECASE,
)
INTEGER_RE = re.compile(r"[+-]?\d[\d,]*")


class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_minutes: float) -> None:
        self.max_seconds = max_minutes * 60
        self.started = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.started >= self.max_seconds:
            control.should_training_stop = True
            control.should_save = True
        return control


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def extract_answer(text: str) -> str | None:
    # Prefer the explicitly committed final answer over earlier boxed or
    # intermediate values, then retain the competition path's integer fallback.
    for pattern in (FINAL_RE, BOXED_RE, ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].replace(",", "")
    matches = INTEGER_RE.findall(text)
    return matches[-1].replace(",", "") if matches else None


def exact_integer_reward(completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        prediction = extract_answer(completion_text(completion))
        normalized = str(expected).strip().replace(",", "")
        rewards.append(1.0 if prediction == normalized else 0.0)
    return rewards


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-minutes", type=float, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--save-steps", type=int, default=None,
                        help="체크포인트 저장 주기 (기본: max_steps//2)")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if args.batch_size % args.num_generations != 0:
        raise SystemExit("--batch-size must be divisible by --num-generations")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = load_dataset("json", data_files=str(args.data), split="train")
    required = {"id", "prompt", "answer"}
    missing = required - set(train_data.column_names)
    if missing:
        raise SystemExit(f"Missing dataset columns: {sorted(missing)}")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(
        base, args.adapter_path, is_trainable=True, local_files_only=True
    )
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        generation_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=2,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps or max(1, args.max_steps // 2),
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        loss_type="dapo",
        scale_rewards="none",
        mask_truncated_completions=True,
        log_completions=True,
        num_completions_to_print=2,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=exact_integer_reward,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        callbacks=[TimeLimitCallback(args.max_minutes)],
    )
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    metadata = {
        "method": "GRPO exact integer reward",
        "base_adapter": str(args.adapter_path),
        "dataset": str(args.data),
        "dataset_sha256": sha256(args.data),
        "samples": len(train_data),
        "global_step": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "max_minutes": args.max_minutes,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "seed": args.seed,
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"adapter={final_dir}")


if __name__ == "__main__":
    main()
