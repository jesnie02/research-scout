import os
from pathlib import Path

import pytest

from scout import config
from scout.config import Settings, load_dotenv, openalex_api_key

KEYS = [
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENALEX_API_KEY",
    "SCOUT_TEST_A",
    "SCOUT_TEST_B",
    "SCOUT_TEST_C",
]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Never read the real .env or the developer's shell keys in tests."""
    for key in KEYS:
        # setenv first so monkeypatch restores the original state after the test,
        # including removing values that load_dotenv() wrote.
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(config, "DOTENV_PATH", env_file)
    return env_file


def test_dotenv_skips_comments_blank_lines_and_empty_values(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_env.write_text(
        '# comment\n\nSCOUT_TEST_A="quoted"\nSCOUT_TEST_B=\nnot a pair\nSCOUT_TEST_C=plain\n'
    )
    load_dotenv()
    assert os.environ["SCOUT_TEST_A"] == "quoted"
    assert "SCOUT_TEST_B" not in os.environ
    assert os.environ["SCOUT_TEST_C"] == "plain"


def test_shell_variables_win_over_dotenv(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCOUT_TEST_A", "from-shell")
    isolated_env.write_text("SCOUT_TEST_A=from-file\n")
    load_dotenv()
    assert os.environ["SCOUT_TEST_A"] == "from-shell"


def test_missing_dotenv_is_fine() -> None:
    load_dotenv()


def test_settings_from_env(isolated_env: Path) -> None:
    isolated_env.write_text(
        "LLM_BASE_URL=https://api.example.com/v1\nLLM_MODEL=some-model\nLLM_API_KEY=secret\n"
    )
    settings = Settings.from_env()
    assert settings.base_url == "https://api.example.com/v1"
    assert settings.model == "some-model"
    assert settings.max_retries == 2


@pytest.mark.parametrize(
    "base_url",
    ["https://api.cohere.com", "https://api.cohere.ai/compatibility/v1"],
)
def test_settings_from_env_use_the_native_cohere_client(isolated_env: Path, base_url: str) -> None:
    # Cohere's OpenAI-compatible API rejects tool_choice (measured 2026-11-17), so any Cohere
    # URL selects PydanticAI's native Cohere model.
    isolated_env.write_text(
        f"LLM_BASE_URL={base_url}\nLLM_MODEL=command-a-plus-05-2026\nLLM_API_KEY=secret\n"
    )
    settings = Settings.from_env()
    assert settings.provider == "cohere"
    assert settings.supports_parallel_tool_calls is False


def test_settings_from_env_take_preset_capabilities_for_a_known_url(isolated_env: Path) -> None:
    isolated_env.write_text(
        "LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/\n"
        "LLM_MODEL=gemini-3.8-flash\nLLM_API_KEY=secret\n"
    )
    settings = Settings.from_env()
    assert settings.provider == "openai_compatible"
    assert settings.supports_parallel_tool_calls is False


def test_other_urls_use_the_openai_compatible_client(isolated_env: Path) -> None:
    isolated_env.write_text(
        "LLM_BASE_URL=https://api.groq.com/openai/v1\nLLM_MODEL=m\nLLM_API_KEY=secret\n"
    )
    settings = Settings.from_env()
    assert settings.provider == "openai_compatible"
    assert settings.supports_parallel_tool_calls is True


def test_key_is_not_shown_in_repr() -> None:
    settings = Settings(model="m", base_url="https://api.example.com/v1", api_key="secret")
    assert "secret" not in repr(settings)


@pytest.mark.parametrize(
    ("base_url", "model", "api_key"),
    [
        ("http://api.example.com/v1", "m", "k"),
        ("https://user:pw@api.example.com/v1", "m", "k"),
        ("https://api.example.com/v1?key=1", "m", "k"),
        ("https://api.example.com/v1", "", "k"),
        ("https://api.example.com/v1", "m", ""),
    ],
)
def test_invalid_settings_are_rejected(base_url: str, model: str, api_key: str) -> None:
    with pytest.raises(ValueError):
        Settings(model=model, base_url=base_url, api_key=api_key)


def test_cohere_preset_uses_its_own_key_and_default_model(isolated_env: Path) -> None:
    isolated_env.write_text("COHERE_API_KEY=ck\n")
    settings = Settings.from_preset("cohere")
    assert settings.provider == "cohere"
    assert settings.base_url == "https://api.cohere.com"
    assert settings.model == "command-a-plus-05-2026"
    assert settings.api_key == "ck"
    assert settings.supports_parallel_tool_calls is False


def test_gemini_preset(isolated_env: Path) -> None:
    isolated_env.write_text("GEMINI_API_KEY=gk\n")
    settings = Settings.from_preset("gemini")
    assert settings.provider == "openai_compatible"
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.model == "gemini-3.8-flash"
    assert settings.api_key == "gk"
    # Not documented for Gemini's compatibility endpoint, so it is not sent.
    assert settings.supports_parallel_tool_calls is False


def test_mistral_is_no_longer_a_preset() -> None:
    assert "mistral-small" not in config.PRESETS


def test_groq_preset_requires_a_model(isolated_env: Path) -> None:
    isolated_env.write_text("GROQ_API_KEY=gk\n")
    with pytest.raises(ValueError):
        Settings.from_preset("groq")
    settings = Settings.from_preset("groq", model="some-groq-model")
    assert settings.model == "some-groq-model"
    assert settings.provider == "openai_compatible"
    assert settings.supports_parallel_tool_calls is True


def test_openalex_key(isolated_env: Path) -> None:
    with pytest.raises(ValueError):
        openalex_api_key()
    isolated_env.write_text("OPENALEX_API_KEY=ok\n")
    assert openalex_api_key() == "ok"
