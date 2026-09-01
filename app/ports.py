from __future__ import annotations

from typing import Literal, Protocol, TypedDict, TypeVar

from pydantic import BaseModel

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class StructuredLLM(Protocol):
    model: str

    async def generate_structured(
        self,
        *,
        messages: list[LLMMessage],
        response_model: type[ResponseModel],
        temperature: float = 0.0,
    ) -> ResponseModel: ...

    async def close(self) -> None: ...
