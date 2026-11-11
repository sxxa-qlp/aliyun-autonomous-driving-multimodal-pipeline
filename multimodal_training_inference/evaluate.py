"""Compare base and LoRA with identical validation ID, image, prompt and generation.

This is a smoke-test / style comparison, NOT a statistically valid evaluation
for 19 nearby training frames and one held-out nearby frame.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from config import settings
from .model_loader import load_base_model
from .prepare_dataset import load_records
from .inference import predict_caption


def compare(baseline_path: Path, validation_file: Path, adapter_dir: Path, output: Path,
            model_dir: Path | None = None) -> dict:
    from peft import PeftModel
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    val = load_records(validation_file)
    if len(val) != 1:
        raise ValueError("This comparison command expects exactly one validation record")
    row = val[0]
    if baseline.get("image_id") and baseline["image_id"] != row["id"]:
        raise ValueError("Saved baseline image ID does not match validation sample")
    if baseline.get("image") and Path(baseline["image"]).resolve() != Path(row["image"]).resolve():
        raise ValueError("Saved baseline image path does not match validation sample")
    if baseline.get("reference_caption") != row["caption"]:
        raise ValueError("Reference caption mismatch")
    if baseline.get("do_sample", False) is not False or baseline.get("max_new_tokens", 128) != 128:
        raise ValueError("Unexpected baseline decoding settings")
    if baseline.get("preprocessing_max_tokens", settings.image_max_tokens) != settings.image_max_tokens:
        raise ValueError("Preprocessing token limits mismatch; keep same config")
    model, processor, _ = load_base_model(model_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    prediction = predict_caption(model, processor, row["image"], baseline["prompt"], 128)
    result = {**baseline, "finetuned_caption": prediction,
              "disclaimer": "One adjacent-frame validation image: qualitative style comparison only"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, text in (("GPT reference", result["reference_caption"]),
                        ("Base Qwen2-VL", result["baseline_caption"]),
                        ("Qwen2-VL + LoRA", result["finetuned_caption"])):
        print("\n" + "=" * 55 + "\n" + label + "\n" + "=" * 55 + "\n" + text)
    print("\nSaved:", output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=settings.dataset_dir / "baseline_prediction.json")
    parser.add_argument("--validation-file", type=Path, default=settings.dataset_dir / "validation.jsonl")
    parser.add_argument("--adapter-dir", type=Path, default=settings.lora_output_dir)
    parser.add_argument("--output", type=Path, default=settings.dataset_dir / "inference_comparison.json")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    compare(args.baseline, args.validation_file, args.adapter_dir, args.output, args.model_dir)

if __name__ == "__main__":
    main()
