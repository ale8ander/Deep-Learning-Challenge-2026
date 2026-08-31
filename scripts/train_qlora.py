import argparse
import hashlib
import json
import time
from pathlib import Path

import bitsandbytes
import datasets
import peft
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/numina_10k_qlora"),
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-minutes", type=float, default=105)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--target-modules",
        choices=("qv", "attention", "all"),
        default="all",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="GPU 메모리가 남을 때 사용. forward 재계산을 없애 약 25~30% 빨라진다.",
    )
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="길이가 비슷한 샘플끼리 배치. 동적 패딩 낭비를 줄여 크게 빨라진다.",
    )
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume Trainer, optimizer, scheduler, and RNG state from a checkpoint directory.",
    )
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="기존 LoRA 어댑터에서 **이어서** 학습한다 (continuation). fresh LoRA 를 만들지 "
             "않고 이 어댑터의 가중치·구성(rank/target)을 그대로 물려받으므로 "
             "--lora-rank/--target-modules 는 무시된다. TIR 특화 continuation 이 "
             "fresh base 실패(CONTEXT 22절)와 구분되는 지점이 바로 이것이다.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise SystemExit(f"Invalid checkpoint directory: {checkpoint}")
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise SystemExit(
                f"Incomplete checkpoint {checkpoint}; missing: {', '.join(missing)}"
            )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("json", data_files=str(args.data), split="train")

    def tokenize(row: dict) -> dict:
        """user/system 턴은 마스킹하고 **모든 assistant 턴**을 학습 대상으로 삼는다.

        예전 구현은 마지막 메시지 하나만 학습했다. 3턴 데이터(system/user/assistant)에서는
        같은 결과지만, TIR처럼 assistant가 여러 번 등장하는 멀티턴 데이터에서는
        **중간 assistant 턴(=파이썬 코드가 들어있는 1라운드)이 통째로 마스킹돼** 학습되지 않았다.
        실제로 그 버그로 TIR SFT 1차 시도가 무효가 됐다(코드 생성률 71%->70%, 무변화).
        """
        messages = row["messages"]
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if len(full_ids) > args.max_seq_length:
            # 초과 샘플은 버린다 (라벨 전부 -100 → 학습 기여 0인 1토큰 더미로 치환).
            # 기존 데이터(전부 상한 이내 검증됨)에는 영향 없음 — R1 장문셋 대응용.
            return {"input_ids": [tokenizer.eos_token_id],
                    "attention_mask": [1], "labels": [-100]}

        labels = [-100] * len(full_ids)
        trained = 0
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prefix = tokenizer.apply_chat_template(
                messages[:index], tokenize=True, add_generation_prompt=True
            )
            upto = tokenizer.apply_chat_template(
                messages[: index + 1], tokenize=True, add_generation_prompt=False
            )
            if full_ids[: len(prefix)] != prefix or full_ids[: len(upto)] != upto:
                raise ValueError("Chat template turn is not a prefix of the full sample")
            labels[len(prefix) : len(upto)] = full_ids[len(prefix) : len(upto)]
            trained += len(upto) - len(prefix)
        if trained == 0:
            raise ValueError("No assistant turn found to train on")

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    tokenized = raw.map(
        tokenize,
        remove_columns=raw.column_names,
        desc="tokenize SFT data",
        num_proc=4,
    )
    _before = len(tokenized)
    tokenized = tokenized.filter(
        lambda r: any(l != -100 for l in r["labels"]), num_proc=4,
        desc="drop overlength")
    if _before != len(tokenized):
        print(f"길이 초과로 제외: {_before - len(tokenized)} / {_before}")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    target_modules = {
        "qv": ["q_proj", "v_proj"],
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }[args.target_modules]
    if args.init_adapter is not None:
        if not (args.init_adapter / "adapter_config.json").is_file():
            raise SystemExit(f"어댑터 아님: {args.init_adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(args.init_adapter), is_trainable=True, local_files_only=True
        )
        print(f"[continuation] {args.init_adapter} 에서 이어서 학습")
    else:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=20,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        group_by_length=args.group_by_length,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    callback = TimeLimitCallback(args.max_minutes)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[callback],
    )

    started = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    elapsed = time.time() - started
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    metadata = {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_path": str(args.model_path),
        "dataset": str(args.data),
        "dataset_sha256": sha256(args.data),
        "samples": len(tokenized),
        "global_step": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "max_minutes": args.max_minutes,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "lora_rank": args.lora_rank,
        "target_modules": args.target_modules,
        "epochs": args.epochs,
        "seed": args.seed,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
        "init_adapter": (
            str(args.init_adapter) if args.init_adapter is not None else None
        ),
        "init_adapter": (
            str(args.init_adapter) if args.init_adapter is not None else None
        ),
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
