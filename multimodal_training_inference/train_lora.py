"""One-GPU Qwen2-VL 4-bit QLoRA SFT; full JSONL inputs; caption-only loss.

Smoke-test recipe for 19 adjacent frames: 1 epoch, r=8, accumulation=4.
No claim of downstream generalization; initial model and image preprocessor frozen.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from PIL import Image
from config import settings
from .model_loader import load_base_model
from .prepare_dataset import load_records
from .inference import PROMPT


def attach_lora_once(base_model, *, rank: int = 8):
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if isinstance(base_model, PeftModel):
        raise ValueError("LoRA is already attached. Do not call get_peft_model twice.")
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.config.use_cache = False
    config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    return model


def encode_training_sample(record: dict, processor, prompt: str = PROMPT):
    """Teacher-forcing: mask user/image prefix to -100 and train caption tokens."""
    import torch
    with Image.open(record["image"]) as im:
        image = im.convert("RGB")
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    assistant = {"role": "assistant", "content": str(record["caption"])}
    prefix_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
    full = processor(text=[full_text], images=[image], return_tensors="pt")
    prefix = processor(text=[prefix_text], images=[image], return_tensors="pt")
    prefix_len = prefix["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :prefix_len], prefix["input_ids"]):
        raise ValueError("Tokenizer prefix mismatch: verify chat template / special tokens")
    labels = full["input_ids"].clone()
    labels[:, :prefix_len] = -100
    if not (labels != -100).any():
        raise ValueError("No supervised caption tokens")
    full["labels"] = labels
    return {key: tensor.to("cuda") for key, tensor in full.items()}


def train(records: list[dict], model, processor, *, epochs: int = 1,
          accumulation: int = 4, learning_rate: float = 1e-4, seed: int = 20260520,
          max_steps: int = 0) -> list[float]:
    import torch
    if not records:
        raise ValueError("Training set is empty")
    if accumulation < 1 or epochs < 1:
        raise ValueError("epochs and accumulation must be >= 1")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    history = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    optimizer_updates = 0
    for epoch in range(epochs):
        shuffled = records.copy()
        random.Random(seed + epoch).shuffle(shuffled)
        for step, record in enumerate(shuffled, 1):
            batch = encode_training_sample(record, processor)
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf training loss at epoch={epoch+1} sample={step}")
            history.append(float(loss.item()))
            start = ((step - 1) // accumulation) * accumulation
            window = min(accumulation, len(shuffled) - start)
            (loss / window).backward()
            if step % accumulation == 0 or step == len(shuffled):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Invalid LoRA gradients")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
            print(f"Epoch {epoch+1}/{epochs} sample {step}/{len(shuffled)} loss={loss.item():.4f}")
            del batch, loss, output
            if max_steps and optimizer_updates >= max_steps:
                break
        if max_steps and optimizer_updates >= max_steps:
            break
    print(f"Optimizer updates: {optimizer_updates}; average loss={sum(history)/len(history):.4f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=settings.dataset_dir / "train.jsonl")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=settings.lora_output_dir)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-optimizer-steps", type=int, default=0, help="Limit optimizer updates for a smoke test (0=all)")
    args = parser.parse_args()
    if (args.output_dir / "adapter_config.json").exists():
        raise RuntimeError(f"Adapter already exists at {args.output_dir}; choose a new --output-dir to avoid overwriting trained weights")
    records = load_records(args.train_file)
    model, processor, path = load_base_model(args.model_dir)
    model = attach_lora_once(model)
    history = train(records, model, processor, epochs=args.epochs, accumulation=args.accumulation,
                    learning_rate=args.learning_rate, seed=settings.seed, max_steps=args.max_optimizer_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    (args.output_dir / "train_summary.json").write_text(json.dumps({
        "model_id": settings.model_id, "model_dir": str(path), "sample_count": len(records),
        "epochs": args.epochs, "accumulation": args.accumulation,
        "lr": args.learning_rate, "seed": settings.seed,
        "prompt": PROMPT, "loss_history": history,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    # Save a compact loss curve independently of Notebook/Kernel state.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
        ax.set(xlabel="Training sample step", ylabel="Cross-entropy loss", title="QLoRA training loss")
        ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "loss_curve.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; loss_history remains in train_summary.json")
    print("LoRA adapter saved:", args.output_dir)

if __name__ == "__main__":
    main()
