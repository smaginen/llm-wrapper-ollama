from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from app.schemas import ChatMessage, Intent

TEMPLATE_DIR = Path(__file__).parent / "templates"

environment = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(default_for_string=False, default=False),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_answer_prompt(
    *, intent: Intent, message: str, history: list[ChatMessage]
) -> str:
    template = environment.get_template("answer.j2")
    return template.render(intent=intent.value, message=message, history=history)
