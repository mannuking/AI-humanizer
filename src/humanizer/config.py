"""Configuration, resolved from the environment with sane defaults.

Every setting can come from the process environment or a ``.env`` file in the
project root. Nothing here reaches the network, so it is safe to import in
tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Runtime configuration for a humanizer session."""

    # --- MiniMax -------------------------------------------------------
    api_key: str = field(default_factory=lambda: _env("MINIMAX_API_KEY") or _env("MINIMAX_AUTH_KEY"))
    base_url: str = field(
        default_factory=lambda: _env("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    )
    # MiniMax-M3 is the safe default: the M2.x family retires on
    # 2026-11-18 and M2 already returns 404.
    model: str = field(default_factory=lambda: _env("MINIMAX_MODEL", "MiniMax-M3"))
    api_style: str = field(default_factory=lambda: _env("MINIMAX_API_STYLE", "openai"))
    timeout: float = field(default_factory=lambda: _env_float("MINIMAX_TIMEOUT", 120.0))
    max_retries: int = field(default_factory=lambda: _env_int("MINIMAX_MAX_RETRIES", 3))

    # --- Rewrite behaviour ---------------------------------------------
    temperature: float = field(default_factory=lambda: _env_float("HUMANIZER_TEMPERATURE", 0.85))
    top_p: float = field(default_factory=lambda: _env_float("HUMANIZER_TOP_P", 0.95))
    max_tokens: int = field(default_factory=lambda: _env_int("HUMANIZER_MAX_TOKENS", 8192))
    max_passes: int = field(default_factory=lambda: _env_int("HUMANIZER_MAX_PASSES", 3))
    chunk_chars: int = field(default_factory=lambda: _env_int("HUMANIZER_CHUNK_CHARS", 6000))
    target_score: float = field(default_factory=lambda: _env_float("HUMANIZER_TARGET_SCORE", 0.25))
    preserve_word_count: bool = field(
        default_factory=lambda: _env_bool("HUMANIZER_PRESERVE_WORD_COUNT", True)
    )
    word_count_tolerance: float = field(
        default_factory=lambda: _env_float("HUMANIZER_WORD_COUNT_TOLERANCE", 0.08)
    )

    # --- Server ----------------------------------------------------------
    streamlit_port: int = field(default_factory=lambda: _env_int("STREAMLIT_PORT", 8501))

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def missing_key_message(self) -> str:
        return (
            "MINIMAX_API_KEY is not set.\n\n"
            "Set it in the project .env file:\n"
            "    MINIMAX_API_KEY=your-key-here\n"
            "or export it in the shell before starting the app."
        )


def load_config() -> Config:
    """Return a Config built from the current environment."""
    return Config()
