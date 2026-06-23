from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COD_SPECTRUM_", env_file=".env")

    database_url: str = f"sqlite:///{ROOT / 'cod_spectrum.db'}"
    data_dir: Path = ROOT / "data"
    sample_fps: float = 1.0
    region_change_threshold: float = 2.0
    hardpoint_target: int = 250
    break_debounce_seconds: float = 2.0
    xmwp_k: float = 0.15
    ocr_engine: str = "stub"
    enable_gpu: bool = False
    max_job_attempts: int = 3


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for directory in ("videos", "frames", "crops", "clips", "reports", "fixtures"):
        (settings.data_dir / directory).mkdir(parents=True, exist_ok=True)
    return settings

