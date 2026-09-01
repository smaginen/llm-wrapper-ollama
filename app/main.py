from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.ollama_client import (
    OllamaClient,
    OllamaResponseError,
    OllamaUnavailableError,
)
from app.schemas import HealthResponse, WrapperRequest, WrapperResponse
from app.service import LLMWrapper


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    app.state.settings = settings
    app.state.llm_client = client
    app.state.wrapper = LLMWrapper(
        llm=client,
        max_history_messages=settings.max_history_messages,
    )
    yield
    await client.close()


app = FastAPI(
    title="Slurm LLM-wrapper",
    version="0.1.0",
    description="Учебный wrapper с intent-классификацией и локальной Ollama.",
    lifespan=lifespan,
)


@app.exception_handler(OllamaUnavailableError)
async def ollama_unavailable_handler(
    _request: Request, error: OllamaUnavailableError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(OllamaResponseError)
async def ollama_response_handler(
    _request: Request, error: OllamaResponseError
) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(error)})


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = cast(Settings, request.app.state.settings)
    return HealthResponse(model=settings.ollama_model)


@app.post("/chat", response_model=WrapperResponse)
async def chat(payload: WrapperRequest, request: Request) -> WrapperResponse:
    wrapper = cast(LLMWrapper, request.app.state.wrapper)
    return await wrapper.handle(payload)
