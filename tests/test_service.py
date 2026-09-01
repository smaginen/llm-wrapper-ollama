from __future__ import annotations

import pytest

from app.schemas import ChatMessage, Intent, WrapperRequest
from app.service import OFF_TOPIC_ANSWER, LLMWrapper
from tests.fakes import FakeLLM


@pytest.mark.asyncio
async def test_course_question_is_classified_and_answered() -> None:
    llm = FakeLLM(
        [
            {
                "intent": "course_question",
                "confidence": 0.97,
                "reason": "Вопрос относится к теории курса.",
            },
            {"answer": "Wrapper управляет отдельными вызовами LLM."},
        ]
    )
    wrapper = LLMWrapper(llm=llm)

    response = await wrapper.handle(WrapperRequest(message="Чем wrapper отличается от агента?"))

    assert response.intent is Intent.COURSE_QUESTION
    assert response.answer == "Wrapper управляет отдельными вызовами LLM."
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_off_topic_does_not_call_answer_generator() -> None:
    llm = FakeLLM(
        [
            {
                "intent": "off_topic",
                "confidence": 0.99,
                "reason": "Запрос не относится к курсу.",
            }
        ]
    )
    wrapper = LLMWrapper(llm=llm)

    response = await wrapper.handle(WrapperRequest(message="Составь меню на неделю"))

    assert response.intent is Intent.OFF_TOPIC
    assert response.answer == OFF_TOPIC_ANSWER
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_only_latest_history_messages_are_rendered() -> None:
    llm = FakeLLM(
        [
            {
                "intent": "practice_help",
                "confidence": 0.9,
                "reason": "Пользователь просит помочь с заданием.",
            },
            {"answer": "Начните со схемы ответа."},
        ]
    )
    wrapper = LLMWrapper(llm=llm, max_history_messages=2)
    history = [
        ChatMessage(role="user", content="старое сообщение"),
        ChatMessage(role="assistant", content="предыдущий ответ"),
        ChatMessage(role="user", content="актуальное уточнение"),
    ]

    await wrapper.handle(WrapperRequest(message="Что делать дальше?", history=history))

    classifier_messages = llm.calls[0]["messages"]
    answer_prompt = llm.calls[1]["messages"][0]["content"]
    assert all(item["content"] != "старое сообщение" for item in classifier_messages)
    assert any(item["content"] == "предыдущий ответ" for item in classifier_messages)
    assert any(item["content"] == "актуальное уточнение" for item in classifier_messages)
    assert "старое сообщение" not in answer_prompt
    assert "предыдущий ответ" in answer_prompt
    assert "актуальное уточнение" in answer_prompt


def test_blank_message_is_rejected() -> None:
    with pytest.raises(ValueError):
        WrapperRequest(message="   ")
