"""Contratos dos modelos: aplicados a TODOS os modelos registados em ``MODELS``.

Um modelo novo fica testado aqui sem escrever testes novos (passo 5 de "Adicionar um
algoritmo", em docs/architecture.md).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_lab.core.foundation.errors import CapabilityError, ModelError, NotFittedError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.tracking.artifacts import ArtifactStore
from ml_lab.core.training.context import RecordingTrainingContext

DOCS = Path(__file__).resolve().parents[2] / "docs" / "algorithms"
TEMPLATE_SECTIONS = (
    "## Intuition",
    "## Essential maths",
    "## Hyperparameters",
    "## Pitfalls",
)
REDUCTION = TaskType.DIMENSIONALITY_REDUCTION

type Data = tuple[NDArray[np.float64], NDArray[Any] | None, NDArray[np.float64]]


def tiny_dataset(task_type: TaskType, seed: int = 0) -> Data:
    """Um dataset mínimo para ``task_type``: features de treino, alvo e features novas."""
    rng = np.random.default_rng(seed)
    if task_type.is_classification:
        n_classes = 2 if task_type is TaskType.BINARY_CLASSIFICATION else 3
        centers = rng.uniform(-4, 4, size=(n_classes, 3))
        y = np.repeat(np.arange(n_classes, dtype=np.int64), 20)
        X = centers[y] + rng.normal(size=(len(y), 3))
        return X, y, centers + rng.normal(scale=0.1, size=centers.shape)
    if task_type is TaskType.REGRESSION:
        X = rng.normal(size=(60, 3))
        return X, X @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.1, size=60), X[:5]
    # Sem alvo: três nuvens bem separadas, para haver estrutura para encontrar.
    centers = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    X = centers[np.repeat(np.arange(3), 20)] + rng.normal(scale=0.8, size=(60, 3))
    return X, None, centers + rng.normal(scale=0.1, size=centers.shape)


def model(name: str, seed: int = 0) -> LabModel[Any]:
    return MODELS.get(name)(seed=seed)


def output(fitted: LabModel[Any], X: NDArray[np.float64]) -> NDArray[Any]:
    """O que o modelo dá: a projeção, na redução de dimensionalidade; senão, as previsões."""
    return fitted.transform(X) if fitted.info.task_types == {REDUCTION} else fitted.predict(X)


def unseen(fitted: LabModel[Any], X: NDArray[np.float64], new: NDArray[np.float64]) -> Any:
    """Os pontos novos, ou os de treino se o modelo não souber lidar com pontos novos."""
    return new if fitted.info.supports_new_data else X


CASES = [
    pytest.param(name, task_type, id=f"{name}-{task_type}")
    for name, model_class in MODELS.items()
    for task_type in sorted(model_class.info.task_types)
]
NAMES = MODELS.names()


def test_there_is_at_least_one_model() -> None:
    assert NAMES


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_trains_on_a_minimal_dataset_of_each_declared_task(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)

    fitted = model(name).fit(X, y)
    points = unseen(fitted, X, new)
    result = output(fitted, points)

    assert fitted.is_fitted
    if task_type is REDUCTION:
        assert result.ndim == 2
        assert len(result) == len(points)
        assert np.isfinite(result).all()
        with pytest.raises(CapabilityError):
            fitted.predict(points)
        return
    assert result.shape == (len(points),)
    if task_type.is_classification:
        assert y is not None
        assert set(result.tolist()) <= set(y.tolist())
    elif task_type is TaskType.CLUSTERING:
        assert (result >= -1).all()
        assert np.issubdtype(result.dtype, np.integer)
    elif task_type is TaskType.ANOMALY_DETECTION:
        assert set(result.tolist()) <= {0, 1}
    else:
        assert np.isfinite(result).all()


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_probabilities_sum_to_one_when_the_model_gives_them(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)
    fitted = model(name).fit(X, y)

    if not fitted.info.supports_proba:
        with pytest.raises(CapabilityError):
            fitted.predict_proba(new)
        return
    probabilities = fitted.predict_proba(new)

    assert len(probabilities) == len(new)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    classes = fitted.classes()
    if task_type.is_classification:
        assert classes is not None
        assert probabilities.shape[1] == len(classes)


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_anomaly_scores_rank_a_faraway_point_first(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)
    fitted = model(name).fit(X, y)

    if task_type is not TaskType.ANOMALY_DETECTION:
        with pytest.raises(CapabilityError):
            fitted.anomaly_scores(new)
        return
    faraway = np.vstack([new, [[40.0, 40.0, 40.0]]])
    scores = fitted.anomaly_scores(faraway)

    assert scores.shape == (len(faraway),)
    assert np.isfinite(scores).all()
    assert scores[-1] > scores[:-1].max()
    assert fitted.predict(faraway)[-1] == 1


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_the_same_seed_gives_the_same_result(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)

    first = model(name, seed=7).fit(X, y)
    second = model(name, seed=7).fit(X, y)
    points = unseen(first, X, new)

    assert np.array_equal(output(first, points), output(second, points))
    assert json.dumps(first.state(), sort_keys=True) == json.dumps(second.state(), sort_keys=True)


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_iterative_models_announce_their_steps(name: str, task_type: TaskType) -> None:
    X, y, _ = tiny_dataset(task_type)
    context = RecordingTrainingContext()

    fitted = model(name).fit(X, y, context)

    if fitted.info.iterative:
        steps = [step for step, _, _ in context.steps]
        assert steps, "um modelo iterativo tem de anunciar pelo menos um passo"
        assert steps == sorted(steps)
        assert steps[0] == 1
    else:
        assert fitted.is_fitted  # aceita o contexto, mesmo sem nada para anunciar


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_iterative_models_stop_when_asked(name: str, task_type: TaskType) -> None:
    X, y, _ = tiny_dataset(task_type)
    context = RecordingTrainingContext(stop_after=3)

    fitted = model(name).fit(X, y, context)

    assert fitted.is_fitted
    if fitted.info.iterative:
        assert len(context.steps) <= 3


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_observed_or_not_the_result_is_the_same(name: str, task_type: TaskType) -> None:
    # Os modelos que só treinam aos bocados para se mostrarem treinam de uma vez quando
    # ninguém está a ver: o modelo final tem de ser igual.
    X, y, new = tiny_dataset(task_type)

    watched = model(name).fit(X, y, RecordingTrainingContext())
    alone = model(name).fit(X, y)
    points = unseen(alone, X, new)

    assert np.allclose(output(watched, points), output(alone, points))


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_snapshots_carry_the_predictions_on_the_probe(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)
    context = RecordingTrainingContext(probe=new)

    fitted = model(name).fit(X, y, context)
    final = fitted.snapshot_state(new)  # o que o Trainer usa no fim, iterativo ou não

    states = [state for _, state in context.snapshots] + [final]
    if not fitted.info.iterative:
        assert context.snapshots == []
    for state in states:
        json.dumps({key: value for key, value in state.items() if key != "probe"})
        if task_type is REDUCTION or not fitted.info.supports_new_data:
            assert "probe" not in state  # não há nada para mostrar numa grelha
        else:
            assert np.asarray(state["probe"]).shape[0] == len(new)


@pytest.mark.parametrize("name", NAMES)
def test_the_step_limit_is_a_hyperparameter_that_caps_the_steps(name: str) -> None:
    info = MODELS.get(name).info
    if info.step_limit is None:
        return
    assert info.iterative, "só os modelos iterativos têm um limite de passos"
    assert info.step_limit in info.hyperparameters.model_fields
    lowest = max(1, info.hyperparameters.form_schema()["properties"][info.step_limit]["minimum"])
    hyperparameters = info.hyperparameters.parse({info.step_limit: lowest})
    task_type = sorted(info.task_types)[0]
    X, y, _ = tiny_dataset(task_type)
    context = RecordingTrainingContext()

    MODELS.get(name)(hyperparameters).fit(X, y, context)

    assert 0 < len(context.steps) <= lowest


@pytest.mark.parametrize("name", NAMES)
def test_default_hyperparameters_are_valid_and_described(name: str) -> None:
    hyperparameters = MODELS.get(name).info.hyperparameters

    defaults = hyperparameters()
    schema = hyperparameters.form_schema()

    assert hyperparameters.parse({}) == defaults
    assert hyperparameters.parse(defaults.model_dump(mode="json")) == defaults
    assert schema["type"] == "object"
    for field, description in schema["properties"].items():
        assert description.get("title"), f"{field} precisa de um título para o formulário"
        assert description.get("description"), f"{field} precisa de uma explicação"
        assert "default" in description


@pytest.mark.parametrize("name", NAMES)
def test_every_model_has_a_didactic_card_with_its_hyperparameters(name: str) -> None:
    info = MODELS.get(name).info
    card = DOCS / info.doc

    assert card.is_file(), f"falta a ficha didática {card}"
    text = card.read_text(encoding="utf-8")
    assert all(section in text for section in TEMPLATE_SECTIONS)
    for field in info.hyperparameters.model_fields:
        assert f"`{field}`" in text, f"a ficha não explica o hiperparâmetro {field}"


@pytest.mark.parametrize("name", NAMES)
def test_the_technical_card_matches_the_registry(name: str) -> None:
    info = MODELS.get(name).info

    assert info.name == name
    assert info.display_name
    assert info.summary
    assert info.task_types
    assert info.family.label
    assert info.implementation.label


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_misuse_is_explained(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)
    fresh = model(name)

    with pytest.raises(NotFittedError):
        output(fresh, new)
    fitted = fresh.fit(X, y)
    with pytest.raises(ModelError, match="features"):
        output(fitted, new[:, :2])


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_the_learned_state_is_plain_json(name: str, task_type: TaskType) -> None:
    X, y, _ = tiny_dataset(task_type)

    state = model(name).fit(X, y).state()

    json.dumps(state)


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_a_saved_model_comes_back_the_same(name: str, task_type: TaskType, tmp_path: Path) -> None:
    # "Guardar experiência" grava o modelo treinado com o joblib (fase 5): tem de dar com
    # todos os modelos, e o que volta a abrir tem de responder o mesmo.
    X, y, new = tiny_dataset(task_type)
    fitted = model(name).fit(X, y)
    files = ArtifactStore(tmp_path)

    files.save("run-a", fitted)
    reopened = files.model("run-a")

    points = unseen(fitted, X, new)
    assert np.array_equal(output(reopened, points), output(fitted, points))
    assert json.dumps(reopened.state(), sort_keys=True) == json.dumps(
        fitted.state(), sort_keys=True
    )
