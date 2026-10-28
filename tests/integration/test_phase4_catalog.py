"""Critério da fase 4: todos os modelos do catálogo clássico funcionam e têm ficha didática.

Os testes de contrato já percorrem os modelos em dados mínimos. Aqui cada um faz o
percurso todo pelos serviços, num dataset a sério da sua tarefa: preparar, treinar,
avaliar e calcular os dados dos gráficos da avaliação. Os ecrãs que a fase 4 completou (a
avaliação, o Explicar e o Laboratório) estão nos testes de ponta a ponta da web.
"""

from typing import Any

import pytest

from ml_lab.core.models.base import MODELS
from ml_lab.core.services import (
    ExperimentConfig,
    Lab,
    ModelSpec,
    Partition,
    PreparationConfig,
    SourceSpec,
    TaskType,
)

IRIS = SourceSpec("sample", {"name": "iris"})
DATA: dict[TaskType, tuple[SourceSpec, PreparationConfig]] = {
    TaskType.BINARY_CLASSIFICATION: (
        SourceSpec("synthetic", {"generator": "moons", "seed": 0, "parameters": {}}),
        PreparationConfig(target="target"),
    ),
    TaskType.MULTICLASS_CLASSIFICATION: (IRIS, PreparationConfig(target="species")),
    TaskType.REGRESSION: (
        SourceSpec("synthetic", {"generator": "regression", "seed": 0, "parameters": {}}),
        PreparationConfig(target="target"),
    ),
    TaskType.CLUSTERING: (
        IRIS,
        PreparationConfig(target="species", task_type=TaskType.CLUSTERING),
    ),
    TaskType.DIMENSIONALITY_REDUCTION: (
        IRIS,
        PreparationConfig(target="species", task_type=TaskType.DIMENSIONALITY_REDUCTION),
    ),
    TaskType.ANOMALY_DETECTION: (
        SourceSpec("synthetic", {"generator": "anomalies", "seed": 0, "parameters": {}}),
        PreparationConfig(target="target", task_type=TaskType.ANOMALY_DETECTION),
    ),
}
CASES = [
    pytest.param(name, task_type, id=f"{name}-{task_type}")
    for name, model in MODELS.items()
    for task_type in sorted(model.info.task_types)
]


@pytest.fixture(scope="module")
def lab() -> Lab:
    return Lab.create()


def test_the_catalog_covers_every_task_with_several_models(lab: Lab) -> None:
    counts = {task: len(lab.catalog.models(task)) for task in TaskType}

    assert counts[TaskType.BINARY_CLASSIFICATION] >= 10
    assert counts[TaskType.REGRESSION] >= 9
    assert counts[TaskType.CLUSTERING] >= 5
    assert counts[TaskType.DIMENSIONALITY_REDUCTION] >= 3
    assert counts[TaskType.ANOMALY_DETECTION] >= 3
    assert all(card.doc.startswith("# ") for card in lab.catalog.models())


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_every_model_goes_the_whole_way_on_real_data(
    lab: Lab, name: str, task_type: TaskType
) -> None:
    source, preparation = DATA[task_type]
    config = ExperimentConfig(
        dataset=source, preparation=preparation, model=ModelSpec(name=name), seed=11
    )

    result = lab.experiments.run(config)
    charts = lab.analysis.charts(result.run_id)

    report = result.reports[Partition.TEST]
    if not MODELS.get(name).info.supports_new_data:  # o t-SNE só avalia o treino
        report = result.reports[Partition.TRAIN]
    assert report.metrics
    assert charts.task_type is task_type
    if task_type is TaskType.DIMENSIONALITY_REDUCTION:
        assert result.embedding is not None
    else:
        assert result.animation is not None
        assert result.primary is not None
        assert result.primary.value is not None


def test_unsupervised_models_never_see_the_reference_column(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    kmeans = MODELS.get("kmeans")
    original = kmeans.fit

    def spy(self: Any, X: Any, y: Any = None, context: Any = None) -> Any:
        seen.append(y)
        return original(self, X, y, context)

    monkeypatch.setattr(kmeans, "fit", spy)
    source, preparation = DATA[TaskType.CLUSTERING]

    result = lab.experiments.run(
        ExperimentConfig(dataset=source, preparation=preparation, model=ModelSpec(name="kmeans"))
    )

    assert seen == [None]
    assert result.reports[Partition.TEST].metric("adjusted_rand").value is not None
