"""Critério da fase 5: fechar a app, reabrir e comparar dois runs.

Pelos serviços: um ``Lab`` treina dois runs e guarda um; outro ``Lab`` com a mesma base de
dados (como reabrir a app) vê os dois no histórico, compara-os e reabre o guardado com o
modelo, a ponto de os gráficos da avaliação voltarem a funcionar. Na interface, o ecrã
Experiências (guardar, comparar, abrir) está nos testes de ponta a ponta da web.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.services import (
    ExperimentConfig,
    ExperimentResult,
    Lab,
    ModelSpec,
    PreparationConfig,
    SourceSpec,
)

IRIS = SourceSpec("sample", {"name": "iris"})


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Uma base de dados e uma pasta de artefactos vazias, só deste teste."""
    monkeypatch.setenv("ML_LAB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ML_LAB_DATABASE_PATH", str(tmp_path / "artifacts" / "ml_lab.db"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def train(lab: Lab, c: float) -> ExperimentResult:
    lab.datasets.load(IRIS)
    return lab.experiments.run(
        ExperimentConfig(
            dataset=IRIS,
            preparation=PreparationConfig(target="species"),
            model=ModelSpec(name="logistic_regression", hyperparameters={"C": c}),
            seed=42,
        )
    )


def test_closing_reopening_and_comparing_two_runs(fresh: Settings) -> None:
    before = Lab.create(fresh)
    strong, weak = train(before, 10.0), train(before, 0.01)
    before.history.save(strong.run_id, name="C alto")

    after = Lab.create(fresh)  # fechar e reabrir a app: só fica o que está no disco
    runs = {entry.run_id: entry for entry in after.history.runs()}
    comparison = after.history.compare([strong.run_id, weak.run_id])
    reopened = after.history.open(strong.run_id)

    assert set(runs) == {strong.run_id, weak.run_id}
    assert (runs[strong.run_id].name, runs[strong.run_id].saved) == ("C alto", True)
    assert not runs[weak.run_id].saved
    assert [difference.label for difference in comparison.differences] == [
        "Hiperparâmetro · C (inverso da regularização)"
    ]
    accuracy = next(metric for metric in comparison.metrics if metric.name == "accuracy")
    assert accuracy.values == (
        strong.main_report.metric("accuracy").value,
        weak.main_report.metric("accuracy").value,
    )
    assert reopened == strong
    assert after.analysis.charts(strong.run_id).roc  # o modelo voltou do ficheiro
