from __future__ import annotations

from collections import deque
from typing import Any

from app.ports import ResponseModel


class FakeLLM:
    model = "fake-model"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[ResponseModel],
        temperature: float = 0.0,
    ) -> ResponseModel:
        self.calls.append(
            {
                "messages": messages,
                "response_model": response_model,
                "temperature": temperature,
            }
        )
        return response_model.model_validate(self._responses.popleft())

    async def close(self) -> None:
        return None
