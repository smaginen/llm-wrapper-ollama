from __future__ import annotations

from app.config import Settings


def test_settings_have_valid_defaults(monkeypatch) -> None:
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT_SECONDS",
        "MAX_HISTORY_MESSAGES",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_model == "qwen3:4b-instruct"
    assert settings.ollama_timeout_seconds == 60.0
    assert settings.max_history_messages == 10
