"""Settings from the shell or a local .env file. Keys never reach model input or logs."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple
from urllib.parse import urlsplit

DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# "cohere" = PydanticAI's native Cohere model. Cohere's OpenAI-compatible API rejects
# tool_choice ("tool_choice is not supported for this model", measured 2026-11-17).
ProviderKind = Literal["openai_compatible", "cohere"]
COHERE_HOSTS = ("cohere.com", "cohere.ai")


class Preset(NamedTuple):
    provider: ProviderKind
    base_url: str
    key_variable: str
    default_model: str | None
    supports_parallel_tool_calls: bool


# Free-tier providers compared in scripts/check_model.py (both on the course's free-LLM list).
PRESETS: dict[str, Preset] = {
    "cohere": Preset(
        "cohere", "https://api.cohere.com", "COHERE_API_KEY", "command-a-plus-05-2026", False
    ),
    "groq": Preset(
        "openai_compatible", "https://api.groq.com/openai/v1", "GROQ_API_KEY", None, True
    ),
    # parallel_tool_calls is not documented for Gemini's compatibility endpoint, so not sent.
    "gemini": Preset(
        "openai_compatible",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.8-flash",
        False,
    ),
}


def provider_for(base_url: str) -> ProviderKind:
    host = urlsplit(base_url).hostname or ""
    return "cohere" if host.endswith(COHERE_HOSTS) else "openai_compatible"


def load_dotenv() -> None:
    """Read KEY=VALUE lines from .env. Variables already set in the shell win."""
    if not DOTENV_PATH.exists():
        return
    for raw in DOTENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if value:
            os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float = 0.2
    timeout: float = 60.0
    max_retries: int = 2
    provider: ProviderKind = "openai_compatible"
    supports_parallel_tool_calls: bool = True

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query:
            raise ValueError("Use a plain HTTPS API base URL.")
        if not self.model:
            raise ValueError("No model set. Set LLM_MODEL or pass --model.")
        if not self.api_key:
            raise ValueError("The API key is empty. Add it to .env (see .env.example).")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        base_url = os.getenv("LLM_BASE_URL", "")
        provider = provider_for(base_url)
        known = [p for p in PRESETS.values() if p.base_url.rstrip("/") == base_url.rstrip("/")]
        parallel = known[0].supports_parallel_tool_calls if known else provider != "cohere"
        return cls(
            model=os.getenv("LLM_MODEL", ""),
            base_url=base_url,
            api_key=os.getenv("LLM_API_KEY", ""),
            provider=provider,
            supports_parallel_tool_calls=parallel,
        )

    @classmethod
    def from_preset(cls, preset: str, model: str | None = None) -> "Settings":
        load_dotenv()
        chosen = PRESETS[preset]
        return cls(
            model=model or chosen.default_model or "",
            base_url=chosen.base_url,
            api_key=os.getenv(chosen.key_variable, ""),
            provider=chosen.provider,
            supports_parallel_tool_calls=chosen.supports_parallel_tool_calls,
        )


def openalex_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENALEX_API_KEY", "")
    if not key:
        raise ValueError(
            "OPENALEX_API_KEY is empty. Create a free key at https://openalex.org "
            "and add it to .env."
        )
    return key
