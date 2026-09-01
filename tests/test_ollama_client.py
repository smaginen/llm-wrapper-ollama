from __future__ import annotations

import json

import httpx
import pytest

from app.ollama_client import OllamaClient, OllamaResponseError
from app.schemas import GeneratedAnswer


@pytest.mark.asyncio
async def test_client_sends_json_schema_to_ollama() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert payload["model"] == "qwen3:4b-instruct"
        assert payload["format"] == GeneratedAnswer.model_json_schema()
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"answer":"ok"}'}},
        )

    client = OllamaClient(
        base_url="http://ollama.test",
        model="qwen3:4b-instruct",
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            response_model=GeneratedAnswer,
        )
    finally:
        await client.close()

    assert result.answer == "ok"


@pytest.mark.asyncio
async def test_invalid_model_json_is_reported() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "not-json"}},
        )

    client = OllamaClient(
        base_url="http://ollama.test",
        model="qwen3:4b-instruct",
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(OllamaResponseError):
            await client.generate_structured(
                messages=[{"role": "user", "content": "test"}],
                response_model=GeneratedAnswer,
            )
    finally:
        await client.close()
