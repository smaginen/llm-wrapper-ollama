from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Intent(str, Enum):
    COURSE_QUESTION = "course_question"
    PRACTICE_HELP = "practice_help"
    OFF_TOPIC = "off_topic"


class ChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message content must not be blank")
        return value


class WrapperRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class IntentClassification(StrictModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=300)


class GeneratedAnswer(StrictModel):
    answer: str = Field(min_length=1, max_length=8_000)


class WrapperResponse(StrictModel):
    intent: Intent
    answer: str
    confidence: float = Field(ge=0, le=1)
    model: str


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    model: str
