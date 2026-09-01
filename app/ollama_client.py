from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from app.ports import LLMMessage, ResponseModel


class OllamaError(RuntimeError):
    """Base error for a failed Ollama interaction."""


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached or timed out."""


class OllamaResponseError(OllamaError):
    """Ollama returned an invalid response."""


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    async def generate_structured(
        self,
        *,
        messages: list[LLMMessage],
        response_model: type[ResponseModel],
        temperature: float = 0.0,
    ) -> ResponseModel:
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": response_model.model_json_schema(),
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as error:
            raise OllamaUnavailableError(
                "Ollama недоступна. Запустите `ollama serve` и проверьте OLLAMA_BASE_URL."
            ) from error
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            raise OllamaResponseError(
                f"Ollama вернула HTTP {error.response.status_code}: {detail}"
            ) from error

        try:
            envelope = response.json()
            content = envelope["message"]["content"]
            data = json.loads(content)
            return response_model.model_validate(data)
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as error:
            raise OllamaResponseError(
                "Ollama вернула ответ, не соответствующий ожидаемой JSON-схеме."
            ) from error

    async def close(self) -> None:
        await self._client.aclose()
