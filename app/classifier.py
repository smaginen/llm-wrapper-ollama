from __future__ import annotations

from app.ports import StructuredLLM
from app.schemas import IntentClassification

CLASSIFIER_SYSTEM_PROMPT = """Ты маршрутизатор запросов учебного помощника.
Выбери ровно один intent:
- course_question: теория и содержание курса по LLM и AI-агентам;
- practice_help: помощь с кодом, домашним заданием или проектом курса;
- off_topic: всё, что не относится к курсу.

Оцени confidence от 0 до 1 и кратко объясни решение. Не отвечай на сам запрос.
"""


async def classify_intent(llm: StructuredLLM, message: str) -> IntentClassification:
    return await llm.generate_structured(
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        response_model=IntentClassification,
        temperature=0.0,
    )
