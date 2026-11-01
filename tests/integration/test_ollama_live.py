"""O Ollama a sério: o modelo configurado pensa à parte e responde no formato pedido.

Salta-se quando o Ollama não está a correr ou não tem o modelo (o resto dos testes usa o
``ScriptedModel``). Para o correr: abre o Ollama, ``ollama pull qwen3:8b`` e
``uv run pytest -m ollama``.
"""

import json

import pytest

from ml_lab.config import Settings
from ml_lab.core.llm import Message, collect
from ml_lab.core.llm.ollama import OllamaServer
from ml_lab.core.services.llm import LanguageModelService

pytestmark = pytest.mark.ollama


def test_the_model_thinks_apart_and_answers_in_the_format_asked() -> None:
    settings = Settings.from_env()
    status = LanguageModelService(settings).status()
    if not status.ready:
        pytest.skip(status.notices[0].message if status.notices else "O Ollama não está pronto.")
    model = OllamaServer(settings.ollama_url).model(settings.ollama_model)
    schema = {
        "type": "object",
        "properties": {"depth": {"type": "integer", "minimum": 1, "maximum": 10}},
        "required": ["depth"],
    }

    reply = collect(
        model.chat(
            [
                Message(
                    "user",
                    "Uma árvore de decisão decora o treino. Que profundidade máxima propões?",
                )
            ],
            think=True,
            schema=schema,
            seed=0,
        )
    )

    assert reply.thinking.strip()  # pensou, e o pensamento veio à parte
    assert 1 <= json.loads(reply.content)["depth"] <= 10
