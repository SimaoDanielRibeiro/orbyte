"""Serviço de análise: gráficos, validação cruzada, explicabilidade e previsões explicadas."""

import numpy as np
import pytest

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.services import (
    ExperimentConfig,
    Lab,
    ModelSpec,
    PreparationConfig,
    SourceSpec,
    TaskType,
)

IRIS = SourceSpec("sample", {"name": "iris"})
LINE = SourceSpec(
    "synthetic", {"generator": "regression", "seed": 0, "parameters": {"n_features": 3}}
)


@pytest.fixture(scope="module")
def lab() -> Lab:
    return Lab.create()


def run(lab: Lab, model: str, source: SourceSpec, preparation: PreparationConfig) -> str:
    config = ExperimentConfig(
        dataset=source, preparation=preparation, model=ModelSpec(name=model), seed=1
    )
    return lab.experiments.run(config).run_id


def test_the_charts_and_the_heavy_analyses_are_remembered(lab: Lab) -> None:
    run_id = run(lab, "logistic_regression", IRIS, PreparationConfig(target="species"))

    first = lab.analysis.cross_validation(run_id)
    again = lab.analysis.cross_validation(run_id)

    assert again is first
    assert lab.analysis.charts(run_id) is lab.analysis.charts(run_id)
    assert lab.analysis.learning_curve(run_id).points


def test_a_linear_prediction_is_the_intercept_plus_the_contributions(lab: Lab) -> None:
    run_id = run(lab, "linear_regression", LINE, PreparationConfig(target="target"))

    explanation = lab.analysis.explain_row(run_id, 0)

    assert explanation.method == "weights"
    assert explanation.base is not None
    total = explanation.base + sum(item.effect for item in explanation.contributions)
    assert float(explanation.prediction.replace(" ", "").replace(",", ".")) == pytest.approx(
        total, abs=1e-3
    )
    effects = [abs(item.effect) for item in explanation.contributions]
    assert effects == sorted(effects, reverse=True)
    assert "intercept" in explanation.how_to_read  # como se lê, para este método


def test_other_models_are_explained_by_replacing_each_feature_with_its_mean(lab: Lab) -> None:
    run_id = run(lab, "random_forest", IRIS, PreparationConfig(target="species"))

    explanation = lab.analysis.explain_row(run_id, 3)

    assert explanation.method == "replace"
    assert len(explanation.contributions) == 4
    assert sum(p for _, p in explanation.probabilities) == pytest.approx(1.0)
    assert "probability" in explanation.effect_label.lower()
    assert "approximation" in explanation.how_to_read


def test_rows_and_tasks_are_checked(lab: Lab) -> None:
    run_id = run(lab, "logistic_regression", IRIS, PreparationConfig(target="species"))
    clustering = run(
        lab, "kmeans", IRIS, PreparationConfig(target="species", task_type=TaskType.CLUSTERING)
    )

    with pytest.raises(ModelError, match="rows"):
        lab.analysis.explain_row(run_id, lab.analysis.test_rows(run_id))
    with pytest.raises(ModelError, match="against a target"):
        lab.analysis.cross_validation(clustering)
    with pytest.raises(ModelError, match="tasks with a target"):
        lab.analysis.explain_row(clustering, 0)
    assert lab.analysis.charts(clustering).silhouette is not None


def test_a_forgotten_run_cannot_be_analysed(lab: Lab) -> None:
    with pytest.raises(ModelError, match="no longer in memory"):
        lab.analysis.charts("run-que-nao-existe")


def test_the_importance_and_dependence_come_from_the_test_rows(lab: Lab) -> None:
    run_id = run(lab, "gradient_boosting", IRIS, PreparationConfig(target="species"))

    importance = lab.analysis.permutation_importance(run_id)
    dependence = lab.analysis.partial_dependence(run_id, "petal_width")

    assert importance.importances[0].feature.startswith("petal")
    assert len(dependence.values) == lab.analysis.test_rows(run_id)
    assert np.isfinite(np.array(dependence.average)).all()
