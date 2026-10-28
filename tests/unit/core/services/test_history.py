"""O histórico de runs: registar sozinho, guardar, abrir, comparar e reproduzir."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.foundation.errors import (
    InvalidParameterError,
    ModelError,
    RunNotFoundError,
    TrackingError,
)
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.services.experiments import ExperimentResult
from ml_lab.core.services.lab import Lab
from ml_lab.core.tracking.store import InMemoryStore, RunEntry


def settings(folder: Path) -> Settings:
    """As definições do processo, com o registo e os artefactos numa pasta só deste teste."""
    return replace(
        get_settings(), artifacts_dir=folder / "artifacts", database_path=folder / "registo.db"
    )


def run(
    lab: Lab, model: str = "logistic_regression", seed: int = 42, **params: Any
) -> ExperimentResult:
    iris = lab.datasets.load_sample("iris")
    config = ExperimentConfig(
        dataset=iris.source,
        preparation=PreparationConfig(target="species"),
        model=ModelSpec(name=model, hyperparameters=params),
        seed=seed,
    )
    return lab.experiments.run(config)


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    return Lab.create(settings(tmp_path), store=InMemoryStore())


def test_every_finished_run_is_recorded_on_its_own(lab: Lab) -> None:
    result = run(lab)

    entry = lab.history.entry(result.run_id)

    assert [run.run_id for run in lab.history.runs()] == [result.run_id]
    assert entry.metrics == result.main_report.values()
    assert entry.primary_metric == "accuracy"
    assert entry.name == "Regressão logística em Iris"
    assert not entry.saved


def test_saving_keeps_the_model_and_the_animation(lab: Lab, tmp_path: Path) -> None:
    result = run(lab, "logistic_regression_gd")

    entry = lab.history.save(result.run_id, name="  Descida do gradiente  ")

    assert (entry.saved, entry.name) == (True, "Descida do gradiente")
    folder = tmp_path / "artifacts" / "runs" / result.run_id
    assert (folder / "model.joblib").is_file()
    assert (folder / "animation.json").is_file()


def test_a_saved_run_reopens_with_everything_after_closing_the_app(tmp_path: Path) -> None:
    before = Lab.create(settings(tmp_path))
    saved = run(before, "logistic_regression_gd")
    only_recorded = run(before, seed=7)
    before.history.save(saved.run_id)

    after = Lab.create(settings(tmp_path))  # outro Lab, a mesma base de dados: reabrir a app
    reopened = after.history.open(saved.run_id)
    recorded = after.history.open(only_recorded.run_id)

    assert reopened == saved
    assert after.analysis.charts(saved.run_id).roc  # o modelo voltou: os gráficos funcionam
    assert recorded.animation is None  # só registado: sem animação nem modelo
    assert recorded.reports == only_recorded.reports
    with pytest.raises(ModelError):
        after.analysis.charts(only_recorded.run_id)


def test_the_layers_of_a_network_are_kept_only_when_saved(tmp_path: Path) -> None:
    # As camadas em cada snapshot pesam como a animação (perto de 1 MB numa rede grande):
    # ficam de fora do registo e só se guardam a pedido, com ela.
    before = Lab.create(settings(tmp_path))
    saved = run(before, "mlp", max_epochs=10)
    only_recorded = run(before, "mlp", seed=7, max_epochs=10)
    before.history.save(saved.run_id)

    after = Lab.create(settings(tmp_path))
    reopened = after.history.open(saved.run_id)
    recorded = after.history.open(only_recorded.run_id)

    assert saved.network is not None
    assert reopened.network == saved.network
    assert (tmp_path / "artifacts" / "runs" / saved.run_id / "network.json").is_file()
    assert recorded.network is None


def test_a_saved_run_whose_model_does_not_open_still_shows(tmp_path: Path) -> None:
    before = Lab.create(settings(tmp_path))
    result = run(before)
    before.history.save(result.run_id)
    (tmp_path / "artifacts" / "runs" / result.run_id / "model.joblib").write_bytes(b"estragado")

    reopened = Lab.create(settings(tmp_path)).history.open(result.run_id)

    assert reopened.notices[0].code == "not_restored"
    assert reopened.reports == result.reports


def test_the_comparison_shows_only_the_choices_that_change(lab: Lab) -> None:
    strong = run(lab, C=10.0)
    weak = run(lab, C=0.001)

    comparison = lab.history.compare([strong.run_id, weak.run_id])

    assert [difference.label for difference in comparison.differences] == [
        "Hiperparâmetro · C (inverso da regularização)"
    ]
    accuracy = next(metric for metric in comparison.metrics if metric.name == "accuracy")
    assert accuracy.values == (strong.primary.value, weak.primary.value)  # type: ignore[union-attr]
    assert comparison.labels[0].startswith("A · ")
    assert "best" in {notice.code for notice in comparison.notices}


def test_the_preparation_choices_are_written_with_their_names(lab: Lab) -> None:
    plain = run(lab)
    cleaned = lab.experiments.run(
        ExperimentConfig(
            dataset=lab.datasets.load_sample("iris").source,
            preparation=PreparationConfig(
                target="species",
                cleaning=(CleaningStep("drop_columns", {"columns": ["sepal_width"]}),),
                split=SplitConfig(strategy=SplitStrategy.STRATIFIED),
            ),
            model=ModelSpec(name="logistic_regression"),
            seed=42,
        )
    )

    comparison = lab.history.compare([plain.run_id, cleaned.run_id])

    differences = {difference.label: difference.values for difference in comparison.differences}
    # O nome da operação e o da estratégia, e não o JSON da configuração nem o nome do campo.
    assert differences["Limpeza"] == ("nenhuma", "Remover colunas (sepal_width)")
    assert differences["Divisão · estratégia"] == ("Aleatória", "Estratificada")


def test_reproducing_a_run_gives_the_same_metrics(lab: Lab) -> None:
    original = run(lab)

    again = lab.history.reproduce(original.run_id).wait(30)
    comparison = lab.history.compare([original.run_id, again.run_id])

    assert again.run_id != original.run_id
    assert comparison.differences == ()
    assert [notice.code for notice in comparison.notices] == ["reproduced"]
    assert all(metric.best is None for metric in comparison.metrics)  # empates: não há melhor


def test_changing_only_the_seed_is_read_as_chance(lab: Lab) -> None:
    first, second = run(lab, seed=1), run(lab, seed=2)

    comparison = lab.history.compare([first.run_id, second.run_id])

    assert [difference.label for difference in comparison.differences] == ["Seed"]
    assert "seed_only" in {notice.code for notice in comparison.notices}


def test_runs_can_be_filtered_renamed_and_deleted(lab: Lab, tmp_path: Path) -> None:
    classifier = run(lab)
    iris = lab.datasets.load_sample("iris")
    clustering = lab.experiments.run(
        ExperimentConfig(
            dataset=iris.source,
            preparation=PreparationConfig(target="species", task_type=TaskType.CLUSTERING),
            model=ModelSpec(name="kmeans"),
            seed=1,
        )
    )
    lab.history.save(clustering.run_id)

    found = lab.history.runs(task_type=TaskType.CLUSTERING)
    renamed = lab.history.rename(classifier.run_id, "Base")
    lab.history.delete(clustering.run_id)

    assert [entry.run_id for entry in found] == [clustering.run_id]
    assert renamed.name == "Base"
    assert [entry.run_id for entry in lab.history.runs()] == [classifier.run_id]
    assert not (tmp_path / "artifacts" / "runs" / clustering.run_id).exists()
    with pytest.raises(InvalidParameterError, match="vazio"):
        lab.history.rename(classifier.run_id, "   ")
    with pytest.raises(RunNotFoundError):
        lab.history.entry(clustering.run_id)


def test_comparing_needs_between_two_and_five_runs(lab: Lab) -> None:
    result = run(lab)

    with pytest.raises(InvalidParameterError, match="entre 2 e 5"):
        lab.history.compare([result.run_id, result.run_id])


class BrokenStore(InMemoryStore):
    """Um registo que falha sempre ao gravar (por exemplo, com o disco cheio)."""

    def add(self, entry: RunEntry, document: Mapping[str, Any]) -> None:
        raise TrackingError("disco cheio")


def test_a_failing_record_does_not_spoil_the_run(tmp_path: Path) -> None:
    lab = Lab.create(settings(tmp_path), store=BrokenStore())

    result = run(lab)

    assert result.primary is not None
    assert lab.history.runs() == ()
