import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import bitsandbytes
import datasets
import peft
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")
SYSTEM_PROMPT = (
    "Solve the math problem carefully. The answer is always an integer. "
    "End your response with exactly: Final answer: <integer>"
)


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


class DPOCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        sequences = []
        labels = []
        for key in ("chosen", "rejected"):
            for feature in features:
                sequences.append({"input_ids": feature[f"{key}_input_ids"]})
                labels.append(feature[f"{key}_labels"])
        padded = self.tokenizer.pad(
            sequences,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        width = padded["input_ids"].shape[1]
        padded_labels = [row + [-100] * (width - len(row)) for row in labels]
        padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        padded["pair_batch_size"] = torch.tensor(len(features), dtype=torch.long)
        return padded


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = shifted_logits.log_softmax(-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum(-1)


class DPOTrainer(Trainer):
    def __init__(self, *args, beta: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        pair_batch_size = int(inputs.pop("pair_batch_size").item())
        policy_outputs = model(**inputs)
        policy_logps = sequence_logps(policy_outputs.logits, labels)
        disable_context = (
            model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        )
        with torch.no_grad(), disable_context:
            reference_outputs = model(**inputs)
            reference_logps = sequence_logps(reference_outputs.logits, labels)
        policy_chosen, policy_rejected = policy_logps.split(pair_batch_size)
        ref_chosen, ref_rejected = reference_logps.split(pair_batch_size)
        logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
        loss = -F.logsigmoid(self.beta * logits).mean()
        return (loss, policy_outputs) if return_outputs else loss


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("json", data_files=str(args.data), split="train")

    def tokenize(row: dict) -> dict:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        result = {}
        for key in ("chosen", "rejected"):
            full_ids = tokenizer.apply_chat_template(
                prompt + [{"role": "assistant", "content": row[key]}],
                tokenize=True,
                add_generation_prompt=False,
            )
            if len(full_ids) > args.max_seq_length:
                raise ValueError(
                    f"{row['id']} {key} exceeds max length: {len(full_ids)}"
                )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"{row['id']} prompt is not a prefix")
            result[f"{key}_input_ids"] = full_ids
            result[f"{key}_labels"] = (
                [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            )
        return result

    tokenized = raw.map(
        tokenize,
        remove_columns=raw.column_names,
        desc="tokenize DPO pairs",
        num_proc=4,
    )
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
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DPOCollator(tokenizer),
        callbacks=[TimeLimitCallback(args.max_minutes)],
        beta=args.beta,
    )
    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    metadata = {
        "method": "DPO",
        "base_adapter": str(args.adapter_path),
        "dataset": str(args.data),
        "dataset_sha256": sha256(args.data),
        "samples": len(tokenized),
        "global_step": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "max_minutes": args.max_minutes,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "epochs": args.epochs,
        "seed": args.seed,
        "metrics": train_result.metrics,
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
