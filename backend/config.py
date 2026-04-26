"""Zentrale Konfiguration via Umgebungsvariablen / .env"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    env: str = "dev"

    ollama_base_url: str = "http://localhost:11434"
    vision_model: str = "llava:7b"
    max_file_size_mb: int = 20
    upload_dir: str = "./uploads"

    database_path: str = "~/.ollama-vision/gallery.db"
    persistent_upload_dir: str = "~/.ollama-vision/images"
    max_batch_size: int = 20

    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]

    # Optional auth hardening.
    # In production set this to a long random value and require it from clients.
    api_key: str = ""

    # Basic per-IP limits (SlowAPI format)
    rate_limit_default: str = "120/minute"
    rate_limit_analyze: str = "30/minute"
    rate_limit_batch: str = "10/minute"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def database_file(self) -> Path:
        p = Path(self.database_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def persistent_upload_path(self) -> Path:
        p = Path(self.persistent_upload_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
