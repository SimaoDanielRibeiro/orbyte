"""A procura escrita à mão que os testes do analista partilham (as ferramentas e o ciclo).

Cinco candidatos com notas redondas, para cada leitura das ferramentas (a distância, o empate, os
folds ganhos, o "decora o treino") se conferir de cabeça. Os testes não importam este ficheiro:
recebem as fábricas pelas fixtures ``facts`` e ``make_candidate``.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from ml_lab.core.automl.analyst_tools import SearchFacts
from ml_lab.core.automl.data_agent import DataDecision
from ml_lab.core.automl.model_agent import SkippedModel
from ml_lab.core.automl.objective import objective_for
from ml_lab.core.automl.search_log import (
    AgentMessage,
    AgentRole,
    Candidate,
    MessageKind,
    Participant,
    ParticipantKind,
    SearchPhase,
)
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import ModelFamily

_NAMES = {
    "logistic_regression": "Logistic regression",
    "knn": "K-nearest neighbours (KNN)",
    "decision_tree": "Decision tree",
    "svm": "SVM",
}


def _candidate(
    number: int,
    model: str,
    mean: float | None,
    std: float | None,
    *,
    train: float | None = None,
    folds: tuple[float | None, ...] = (),
    family: ModelFamily = ModelFamily.TREE,
    changes: dict[str, Any] | None = None,
    phase: SearchPhase = SearchPhase.SCREENING,
    **extra: Any,
) -> Candidate:
    """Um candidato medido (ou que falhou, sem ``mean``), com 1,25 s nos folds todos."""
    return Candidate(
        number=number,
        phase=phase,
        family=family,
        model=model,
        model_display_name=_NAMES[model],
        hyperparameters={"max_depth": 5, **(changes or {})},
        changes=changes or {},
        mean=mean,
        std=std,
        train_mean=train,
        folds=folds,
        seconds=1.25,
        **extra,
    )


_CANDIDATES = (
    _candidate(
        1, "logistic_regression", 0.90, 0.03, train=0.91, family=ModelFamily.LINEAR,
        folds=(0.87, 0.90, 0.93),
    ),
    _candidate(
        2, "knn", 0.88, 0.05, train=0.90, family=ModelFamily.INSTANCE_BASED,
        folds=(0.83, 0.88, 0.93),
    ),
    _candidate(3, "decision_tree", 0.92, 0.02, train=1.0, folds=(0.90, 0.92, 0.94)),
    _candidate(
        4, "decision_tree", 0.93, 0.02, train=0.95, folds=(0.92, 0.93, 0.94),
        changes={"max_depth": 3}, phase=SearchPhase.TUNING, ai=True,
        reason="The tree memorises: a shallower one should generalise better.",
    ),
    _candidate(5, "svm", None, None, family=ModelFamily.KERNEL, error="It ran out of memory."),
)  # fmt: skip

_MESSAGES = (
    AgentMessage(
        number=1,
        role=AgentRole.AI,
        author="Agent · AI",
        text="I measured Decision tree with maximum depth 3 (default 5): 93.0% ± 2.0%.",
        phase=SearchPhase.TUNING,
        candidate=4,
        kind=MessageKind.REPLY,
        recipients=(Participant(ParticipantKind.COORDINATOR),),
    ),
    AgentMessage(
        number=2,
        role=AgentRole.COORDINATOR,
        author="Coordinator",
        text="On the test: 94.0%. The cross-validation expected 93.0% ± 2.0%. As expected.",
        phase=SearchPhase.FINAL,
        candidate=4,
        kind=MessageKind.TEST,
    ),
)

_FACTS = SearchFacts(
    dataset="Iris",
    target="species",
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    objective=objective_for(TaskType.MULTICLASS_CLASSIFICATION),
    candidates=_CANDIDATES,
    ranking=(4, 3, 1, 2),
    winner=4,
    baseline=1,
    messages=_MESSAGES,
    settings={4: "maximum depth 3 (default 5)"},
    test="94.0%",
    decisions=(
        DataDecision("Drop duplicate rows", "One row repeats another."),
        DataDecision("Task: multiclass classification", "species has three values."),
    ),
    adopted=False,
    train_rows=119,
    test_rows=30,
    search_rows=100,
    folds=3,
    budget=6,
    finalists=2,
    agents=("Agent · Linear models", "Agent · Decision trees"),
    skipped=(
        SkippedModel("lof", "LOF", ModelFamily.INSTANCE_BASED, "it only knows its rows"),
        SkippedModel("tsne", "t-SNE", ModelFamily.LINEAR, "it only knows its rows"),
    ),
    ai_model="qwen3:8b",
    readings=("The winner beats the baseline by 3.0 points.",),
)


@pytest.fixture
def facts() -> Callable[..., SearchFacts]:
    """A procura acabada; os argumentos trocam campos (``facts(winner=None)``)."""

    def make(**changes: Any) -> SearchFacts:
        return replace(_FACTS, **changes)

    return make


@pytest.fixture
def make_candidate() -> Callable[..., Candidate]:
    """A fábrica de candidatos da procura (o número, o modelo, a média e o desvio padrão)."""
    return _candidate
