"""Save before/after captions using exactly one image, Prompt and generation config."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
from config import settings
from .model_loader import load_base_model
from .prepare_dataset import load_records

PROMPT = (
    "Describe this autonomous driving scene in one or two concise "
    "English sentences. Mention the road layout, traffic participants, "
    "weather and clearly visible risks. Do not use bullet points or speculate."
)


def predict_caption(model, processor, image_path: str | Path, prompt: str = PROMPT, max_new_tokens: int = 128) -> str:
    import torch
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt},
    ]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to("cuda")
    model.eval()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def run_baseline(validation_path: Path, baseline_path: Path, model_dir: Path | None = None) -> dict:
    import torch
    samples = load_records(validation_path)
    if not samples:
        raise ValueError("Validation JSONL is empty")
    sample = samples[0]
    if baseline_path.exists():
        old = json.loads(baseline_path.read_text(encoding="utf-8"))
        if old.get("image_id") != sample["id"] or old.get("prompt") != PROMPT:
            raise ValueError("Existing baseline mismatches validation image or PROMPT. Use a new result filename.")
        print("Baseline already saved; reuse:", baseline_path)
        return old
    model, processor, _ = load_base_model(model_dir)
    prediction = predict_caption(model, processor, sample["image"], PROMPT)
    result = {
        "image_id": sample["id"], "image": sample["image"],
        "prompt": PROMPT, "reference_caption": sample["caption"],
        "baseline_caption": prediction, "max_new_tokens": 128,
        "do_sample": False, "preprocessing_max_tokens": settings.image_max_tokens,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Baseline caption:", prediction)
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", type=Path, default=settings.dataset_dir / "validation.jsonl")
    parser.add_argument("--output", type=Path, default=settings.dataset_dir / "baseline_prediction.json")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    run_baseline(args.validation_file, args.output, args.model_dir)

if __name__ == "__main__":
    main()
