"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _numbered_keys(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(_env(f"{prefix}_{index}") for index in range(1, count + 1))


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    supabase_scope_id: str
    enable_supabase_persistence: bool
    groq_api_keys: tuple[str, ...]
    gemini_api_keys: tuple[str, ...]
    openrouter_api_key: str
    provider_timeout_seconds: float
    max_upload_bytes: int
    max_pdf_pages: int

    @property
    def supabase_enabled(self) -> bool:
        return bool(
            self.enable_supabase_persistence
            and self.supabase_url
            and self.supabase_key
            and self.supabase_scope_id
        )

    @property
    def any_text_provider_enabled(self) -> bool:
        return any(self.groq_api_keys) or any(self.gemini_api_keys) or bool(self.openrouter_api_key)



def load_settings() -> Settings:
    """Load and normalize settings once at application startup."""
    timeout_raw = _env("PROVIDER_TIMEOUT_SECONDS", "60")
    upload_raw = _env("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))
    pages_raw = _env("MAX_PDF_PAGES", "6")

    try:
        timeout = max(5.0, min(float(timeout_raw), 120.0))
    except ValueError:
        timeout = 60.0

    try:
        max_upload_bytes = max(1024, min(int(upload_raw), 25 * 1024 * 1024))
    except ValueError:
        max_upload_bytes = 5 * 1024 * 1024

    try:
        max_pdf_pages = max(1, min(int(pages_raw), 20))
    except ValueError:
        max_pdf_pages = 6

    return Settings(
        supabase_url=_env("SUPABASE_URL").rstrip("/"),
        supabase_key=_env("SUPABASE_KEY"),
        supabase_scope_id=_env("SUPABASE_SCOPE_ID"),
        enable_supabase_persistence=_env_bool("ENABLE_SUPABASE_PERSISTENCE"),
        groq_api_keys=_numbered_keys("GROQ_API_KEY", 3),
        gemini_api_keys=_numbered_keys("GEMINI_API_KEY", 4),
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        provider_timeout_seconds=timeout,
        max_upload_bytes=max_upload_bytes,
        max_pdf_pages=max_pdf_pages,
    )


settings = load_settings()
