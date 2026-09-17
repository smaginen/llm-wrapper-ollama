from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.main import app
from app.ollama_client import OllamaResponseError, OllamaUnavailableError
from app.ports import LLMMessage, ResponseModel
from app.service import LLMWrapper
from tests.fakes import FakeLLM


@pytest.mark.asyncio
async def test_health_and_chat_contracts() -> None:
    llm = FakeLLM(
        [
            {
                "intent": "course_question",
                "confidence": 0.95,
                "reason": "Вопрос относится к курсу.",
            },
            {"answer": "Wrapper выполняет заранее заданный pipeline."},
        ]
    )
    app.state.settings = Settings(ollama_model=llm.model)
    app.state.wrapper = LLMWrapper(llm=llm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health_response = await client.get("/health")
        chat_response = await client.post(
            "/chat",
            json={"message": "Чем wrapper отличается от агента?", "history": []},
        )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "model": "fake-model"}
    assert chat_response.status_code == 200
    assert chat_response.json() == {
        "intent": "course_question",
        "answer": "Wrapper выполняет заранее заданный pipeline.",
        "confidence": 0.95,
        "model": "fake-model",
    }


class FailingLLM:
    model = "failing-model"

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate_structured(
        self,
        *,
        messages: list[LLMMessage],
        response_model: type[ResponseModel],
        temperature: float = 0.0,
    ) -> ResponseModel:
        raise self._error

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (OllamaUnavailableError("Ollama недоступна"), 503),
        (OllamaResponseError("Некорректный ответ Ollama"), 502),
    ],
)
async def test_llm_errors_are_exposed_as_gateway_errors(
    error: Exception, expected_status: int
) -> None:
    app.state.wrapper = LLMWrapper(llm=FailingLLM(error))

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "Помоги с практикой"})

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(error)}
