from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class StructuredLLM(Protocol):
    model: str

    async def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[ResponseModel],
        temperature: float = 0.0,
    ) -> ResponseModel: ...

    async def close(self) -> None: ...
