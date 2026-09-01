from __future__ import annotations

import asyncio

from app.config import Settings
from app.ollama_client import OllamaClient
from app.schemas import WrapperRequest
from app.service import LLMWrapper


async def main() -> None:
    settings = Settings.from_env()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    try:
        wrapper = LLMWrapper(
            llm=client,
            max_history_messages=settings.max_history_messages,
        )
        result = await wrapper.handle(
            WrapperRequest(message="Как сделать LLM-wrapper для домашнего задания?")
        )
        print(result.model_dump_json(indent=2))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
