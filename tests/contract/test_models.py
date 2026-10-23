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
from ml_lab.core.training.context import RecordingTrainingContext

DOCS = Path(__file__).resolve().parents[2] / "docs" / "algorithms"
TEMPLATE_SECTIONS = (
    "## Intuição",
    "## Matemática essencial",
    "## Hiperparâmetros",
    "## Armadilhas",
)

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
    X = rng.normal(size=(60, 3))
    if task_type is TaskType.REGRESSION:
        return X, X @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.1, size=60), X[:5]
    return X, None, X[:5]


def model(name: str, seed: int = 0) -> LabModel[Any]:
    return MODELS.get(name)(seed=seed)


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
    predictions = fitted.predict(new)

    assert fitted.is_fitted
    assert predictions.shape == (len(new),)
    if task_type.is_classification:
        assert y is not None
        assert set(predictions.tolist()) <= set(y.tolist())
    else:
        assert np.isfinite(predictions).all()


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_probabilities_sum_to_one_when_the_model_gives_them(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)
    fitted = model(name).fit(X, y)

    if not fitted.info.supports_proba or not task_type.is_classification:
        with pytest.raises(CapabilityError):
            fitted.predict_proba(new)
        return
    probabilities = fitted.predict_proba(new)
    classes = fitted.classes()

    assert classes is not None
    assert probabilities.shape == (len(new), len(classes))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_the_same_seed_gives_the_same_result(name: str, task_type: TaskType) -> None:
    X, y, new = tiny_dataset(task_type)

    first = model(name, seed=7).fit(X, y)
    second = model(name, seed=7).fit(X, y)

    assert np.array_equal(first.predict(new), second.predict(new))
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
        fresh.predict(new)
    fitted = fresh.fit(X, y)
    with pytest.raises(ModelError, match="features"):
        fitted.predict(new[:, :2])


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_the_learned_state_is_plain_json(name: str, task_type: TaskType) -> None:
    X, y, _ = tiny_dataset(task_type)

    state = model(name).fit(X, y).state()

    json.dumps(state)
