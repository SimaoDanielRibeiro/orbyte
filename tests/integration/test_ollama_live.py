"""O Ollama a sério: o modelo configurado pensa à parte, responde no formato pedido e, como
analista, explica o vencedor de uma procura usando as ferramentas.

Salta-se quando o Ollama não está a correr ou não tem o modelo (o resto dos testes usa o
``ScriptedModel``). Para o correr: abre o Ollama, ``ollama pull qwen3:8b`` e
``uv run pytest -m ollama``.
"""

import json

import pytest

from ml_lab.config import Settings
from ml_lab.core.llm import Message, collect
from ml_lab.core.llm.ollama import OllamaServer
from ml_lab.core.services import (
    AnalystRequest,
    AnalystStepKind,
    AutoMLRequest,
    Lab,
    SearchSettings,
    SourceSpec,
)
from ml_lab.core.services.llm import LanguageModelService
from ml_lab.core.tracking.store import InMemoryStore

pytestmark = pytest.mark.ollama


def ready_settings() -> Settings:
    """As definições do processo, se o Ollama estiver pronto; senão, o teste salta."""
    settings = Settings.from_env()
    status = LanguageModelService(settings).status()
    if not status.ready:
        pytest.skip(status.notices[0].message if status.notices else "O Ollama não está pronto.")
    return settings


def test_the_model_thinks_apart_and_answers_in_the_format_asked() -> None:
    settings = ready_settings()
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


def test_the_analyst_explains_the_winner_with_its_tools() -> None:
    lab = Lab.create(ready_settings(), store=InMemoryStore())
    result = lab.automl.run(
        AutoMLRequest(
            dataset=SourceSpec("sample", {"name": "iris"}),
            target="species",
            models=("logistic_regression", "knn", "decision_tree"),
            settings=SearchSettings(budget=6, folds=3, finalists=2),
            seed=3,
        )
    )
    winner = result.winner_candidate
    assert winner is not None

    answer = lab.automl.ask(
        result.search_id, AnalystRequest(question="Why did the winner win?")
    ).wait(timeout=600)

    used = [step.tool.name for step in answer.trace if step.tool is not None]
    assert "list_candidates" in used  # leu o quadro antes de responder
    assert any(step.kind is AnalystStepKind.THOUGHT for step in answer.trace)
    named = (f"#{winner.number}", winner.model_display_name.lower())
    assert any(name in answer.answer.lower() for name in named)  # fala do vencedor
    # As respostas ficaram no pedido da resposta: repeti-lo dá o mesmo, sem perguntar ao Ollama.
    again = lab.automl.ask(result.search_id, answer.request).wait(timeout=60)
    assert (again.answer, again.model) == (answer.answer, "saved answers")
    assert [step.text for step in again.trace] == [step.text for step in answer.trace]
