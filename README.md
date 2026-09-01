# LLM-wrapper на локальной Ollama

Учебный проект по первому заданию курса «ИИ-агенты»:

```text
request -> intent classification -> Jinja2 prompt -> Ollama -> Pydantic response
```

По умолчанию используется уже установленная модель `qwen3:4b-instruct`. Все запросы выполняются локально через Ollama; внешний API key не нужен.

## Возможности

- FastAPI endpoints `/health` и `/chat`;
- три intent: `course_question`, `practice_help`, `off_topic`;
- отдельный структурированный LLM-вызов для классификации;
- короткая история диалога в Jinja2-промпте;
- JSON Schema передаётся Ollama через поле `format`;
- ответ проверяется Pydantic-моделью;
- off-topic запрос не запускает второй LLM-вызов;
- сетевые и невалидные ответы превращаются в понятные HTTP 502/503;
- unit-тесты не обращаются к реальной модели.

## Запуск

Требуется Python 3.10+ и запущенная Ollama.

```bash
cd llm-wrapper-ollama
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ollama serve
```

В другом терминале:

```bash
cd llm-wrapper-ollama
source .venv/bin/activate
uvicorn app.main:app --reload
```

Swagger UI: <http://127.0.0.1:8000/docs>

## Пример запроса

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Чем LLM-wrapper отличается от агента?",
    "history": []
  }'
```

Продолжение диалога передаётся клиентом явно:

```json
{
  "message": "А когда в нём появляется agent loop?",
  "history": [
    {"role": "user", "content": "Чем wrapper отличается от агента?"},
    {"role": "assistant", "content": "Wrapper управляет отдельными вызовами LLM."}
  ]
}
```

## Настройка

Переменные окружения перечислены в `.env.example`. Например:

```bash
export OLLAMA_MODEL=qwen3:8b
export OLLAMA_TIMEOUT_SECONDS=90
```

Файл `.env` автоматически не читается: переменные нужно экспортировать в shell или подключить менеджер окружения самостоятельно.

## Проверки

```bash
pytest
ruff check .
python scripts/smoke_test.py
```

## Что объяснить на защите

1. Wrapper управляет фиксированным pipeline; агент самостоятельно повторяет цикл выбора tools.
2. Pydantic гарантирует форму результата после валидации, но не истинность ответа.
3. Jinja2 отделяет текст промпта от orchestration-кода.
4. История приходит в запросе: сервер пока не хранит долговременную память.
5. Два LLM-вызова дороже одного, зато classifier становится отдельным тестируемым шагом.
