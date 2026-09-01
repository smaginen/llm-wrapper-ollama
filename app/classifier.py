from __future__ import annotations

from app.ports import LLMMessage, StructuredLLM
from app.schemas import ChatMessage, IntentClassification

CLASSIFIER_SYSTEM_PROMPT = """Ты маршрутизатор запросов учебного помощника.
Выбери ровно один intent:
- course_question: теория и содержание курса по LLM и AI-агентам;
- practice_help: помощь с кодом, домашним заданием или проектом курса;
- off_topic: всё, что не относится к курсу.

Оцени confidence от 0 до 1 и кратко объясни решение. Не отвечай на сам запрос.
"""


async def classify_intent(
    llm: StructuredLLM,
    message: str,
    history: list[ChatMessage] | None = None,
) -> IntentClassification:
    context: list[LLMMessage] = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
    ]
    context.extend(
        {"role": item.role, "content": item.content} for item in (history or [])
    )
    context.append({"role": "user", "content": message})
    return await llm.generate_structured(
        messages=context,
        response_model=IntentClassification,
        temperature=0.0,
    )
