"""Central settings for the DSW-based, two-stage autonomous-driving demo.

Import `settings` anywhere. Read secrets from environment or local `.env` only;
never commit the latter. `python config.py --check` checks cloud and GPU setup.
"""
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    region: str = os.getenv("ALIYUN_REGION", "cn-hangzhou")
    dlf_rest_uri: str = os.getenv("DLF_REST_URI", "https://dlfnext.cn-hangzhou.aliyuncs.com")
    dlf_warehouse: str = os.getenv("DLF_WAREHOUSE", "vehicle_lake_demo")
    oss_endpoint: str = os.getenv("DLF_OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
    access_key_id: str = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
    access_key_secret: str = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
    security_token: str = os.getenv("ALIBABA_CLOUD_SECURITY_TOKEN", "")
    raw_table: str = os.getenv("PAIMON_RAW_TABLE", "default.camfront_raw_images")
    enriched_table: str = os.getenv("PAIMON_ENRICHED_TABLE", "default.camfront_ai_enriched")
    vision_base_url: str = os.getenv("VISION_API_BASE_URL", "")
    vision_key: str = os.getenv("VISION_API_KEY", "")
    vision_model: str = os.getenv("VISION_MODEL", "your-vision-model")
    embedding_url: str = os.getenv("EMBEDDING_API_URL", "")
    embedding_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "tongyi-embedding-vision-plus")
    vector_dim: int = int(os.getenv("EMBEDDING_VECTOR_DIM", "1152"))
    work_dir: Path = Path(os.getenv("WORK_DIR", "/mnt/systemDisk/ad_multimodal"))
    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
    model_cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", "/mnt/systemDisk/model_cache"))
    seed: int = int(os.getenv("SPLIT_RANDOM_SEED", "20260520"))
    val_ratio: float = float(os.getenv("VALIDATION_RATIO", "0.05"))
    image_max_tokens: int = int(os.getenv("IMAGE_MAX_TOKENS", "256"))

    @property
    def dataset_dir(self) -> Path:
        return self.work_dir / "vlm_training"

    @property
    def lora_output_dir(self) -> Path:
        return self.work_dir / "qwen2vl_lora"

    @property
    def nu_scenes_dir(self) -> Path:
        return self.work_dir / "nuscenes"

    @property
    def caption_csv(self) -> Path:
        return self.work_dir / "caption_metadata_results.csv"

    @property
    def embedding_cache(self) -> Path:
        return self.work_dir / "embedding_cache" / self.embedding_model

    def require_dlf_credentials(self) -> None:
        if not self.access_key_id or not self.access_key_secret:
            raise RuntimeError("Set ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET (or adapt a supported role credential provider).")

    def catalog_options(self) -> dict[str, str]:
        """DLF 2.0 REST catalog options used in the verified PyPaimon 1.4 flow."""
        self.require_dlf_credentials()
        options = {
            "metastore": "rest", "uri": self.dlf_rest_uri,
            "warehouse": self.dlf_warehouse,
            "dlf.region": self.region,
            "token.provider": "dlf",
            "dlf.access-key-id": self.access_key_id,
            "dlf.access-key-secret": self.access_key_secret,
            "dlf.oss-endpoint": self.oss_endpoint,
        }
        if self.security_token:
            # Validate against the deployed PyPaimon version before using STS.
            options["dlf.security-token"] = self.security_token
        return options


settings = Settings()


def check_environment() -> bool:
    """Print diagnostics without revealing secrets or changing packages."""
    import importlib.metadata
    import platform
    print("Python:", platform.python_version())
    for name in ("numpy", "pandas", "pypaimon", "pyarrow", "torch", "transformers", "peft", "bitsandbytes", "modelscope"):
        try:
            print(f"{name:15} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name:15} [not installed]")
    print("Work dir:", settings.work_dir)
    print("Work-dir parent exists:", settings.work_dir.parent.exists())
    if settings.work_dir.parent.exists():
        print("Free GiB:", round(shutil.disk_usage(settings.work_dir.parent).free / 2**30, 1))
    for label, value in (("DLF credentials", bool(settings.access_key_id and settings.access_key_secret)),
                         ("Vision API", bool(settings.vision_base_url and settings.vision_key)),
                         ("Embedding API", bool(settings.embedding_url and settings.embedding_key))):
        print(f"{label}: {'configured' if value else 'NOT configured'}")
    try:
        import torch
        print("PyTorch CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            sm = torch.cuda.get_device_capability(0)
            print("GPU:", name, "compute capability:", sm)
            x = torch.randn(64, 64, device="cuda")
            torch.cuda.synchronize()
            print("GPU tensor test:", float((x @ x).mean()))
            return True
    except (ImportError, RuntimeError) as exc:
        print("GPU check failed:", str(exc)[:300])
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check packages, credentials, storage and CUDA")
    args = parser.parse_args()
    if args.check:
        check_environment()
    else:
        parser.print_help()
