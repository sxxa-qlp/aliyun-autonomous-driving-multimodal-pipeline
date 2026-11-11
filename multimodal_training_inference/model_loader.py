"""Load Qwen2-VL-2B-Instruct as NF4 + FP16 on DSW V100 (no FlashAttention2)."""
from __future__ import annotations
from pathlib import Path
from config import settings


def require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; check NVIDIA GPU and PyTorch cu126 installation")
    print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda,
          "architectures:", torch.cuda.get_arch_list())


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    """Prefer explicit local dir; otherwise download/reuse ModelScope cache."""
    if model_dir:
        path = Path(model_dir).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"No config.json under {path}")
        return path
    from modelscope import snapshot_download
    cached = Path(snapshot_download(settings.model_id, cache_dir=str(settings.model_cache_dir)))
    print("ModelScope model directory:", cached)
    return cached


def load_base_model(model_dir: str | Path | None = None, *, max_tokens: int | None = None):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    require_cuda()
    path = resolve_model_dir(model_dir)
    tokens = max_tokens or settings.image_max_tokens
    if tokens < 128:
        print("Warning: very low visual token limit may lose scene details")
    processor = AutoProcessor.from_pretrained(
        path,
        min_pixels=min(128, tokens) * 28 * 28,
        max_pixels=tokens * 28 * 28,
        use_fast=False,
        local_files_only=True,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        path, quantization_config=quantization,
        torch_dtype=torch.float16, device_map={"": 0},
        attn_implementation="eager", local_files_only=True,
    )
    print("Allocated GiB:", round(torch.cuda.memory_allocated() / 2**30, 2))
    return model, processor, path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    model, processor, path = load_base_model(args.model_dir)
    print("Ready:", path, type(model).__name__, type(processor).__name__)

if __name__ == "__main__":
    main()
