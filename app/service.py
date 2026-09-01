from __future__ import annotations

from app.classifier import classify_intent
from app.ports import StructuredLLM
from app.prompt_renderer import render_answer_prompt
from app.schemas import GeneratedAnswer, Intent, WrapperRequest, WrapperResponse

OFF_TOPIC_ANSWER = (
    "Я отвечаю только на вопросы по курсу «ИИ-агенты» и его практическим заданиям."
)


class LLMWrapper:
    def __init__(self, *, llm: StructuredLLM, max_history_messages: int = 10) -> None:
        self._llm = llm
        self._max_history_messages = max_history_messages

    async def handle(self, request: WrapperRequest) -> WrapperResponse:
        history = request.history[-self._max_history_messages :]
        classification = await classify_intent(self._llm, request.message, history)

        if classification.intent is Intent.OFF_TOPIC:
            return WrapperResponse(
                intent=classification.intent,
                answer=OFF_TOPIC_ANSWER,
                confidence=classification.confidence,
                model=self._llm.model,
            )

        prompt = render_answer_prompt(
            intent=classification.intent,
            message=request.message,
            history=history,
        )
        generated = await self._llm.generate_structured(
            messages=[{"role": "user", "content": prompt}],
            response_model=GeneratedAnswer,
            temperature=0.2,
        )
        return WrapperResponse(
            intent=classification.intent,
            answer=generated.answer,
            confidence=classification.confidence,
            model=self._llm.model,
        )
