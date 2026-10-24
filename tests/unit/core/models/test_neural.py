"""A rede neuronal (MLP) em PyTorch: aprende formas curvas, pára a tempo e mostra as camadas."""

from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.models.neural.mlp import MLPClassifier, MLPParams
from ml_lab.core.training.context import RecordingTrainingContext


def make(name: str, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    return model_class(model_class.info.hyperparameters.parse(values), seed=0)


def moons(n: int = 400, noise: float = 0.1) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, np.pi, size=n)
    upper = np.arange(n) < n // 2
    x = np.where(upper, np.cos(angles), 1 - np.cos(angles))
    y = np.where(upper, np.sin(angles), 0.5 - np.sin(angles))
    X = np.column_stack([x, y]) + rng.normal(scale=noise, size=(n, 2))
    return (X - X.mean(axis=0)) / X.std(axis=0), (~upper).astype(np.int64)


def test_the_network_learns_the_curved_boundary_of_the_moons() -> None:
    X, y = moons()

    accuracy = float(np.mean(make("mlp").fit(X, y).predict(X) == y))
    straight = float(np.mean(make("logistic_regression").fit(X, y).predict(X) == y))

    assert accuracy > 0.97
    assert accuracy > straight + 0.05  # uma reta não chega lá


def test_the_regression_network_learns_a_curve_in_the_original_scale() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, size=(400, 1))
    y = 100 + 50 * np.sin(X[:, 0])  # alvo longe de zero: a rede treina-o padronizado

    predicted = make("mlp_regression", units=32).fit(X, y).predict(X)

    r2 = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.95
    assert abs(predicted.mean() - 100) < 5


def test_early_stopping_keeps_the_weights_of_the_best_epoch() -> None:
    X, y = moons(noise=0.3)
    context = RecordingTrainingContext()

    model = make("mlp", patience=5, max_epochs=500).fit(X, y, context)

    state = model.state()
    validation = [metrics["val_loss"] for _, _, metrics in context.steps]
    assert state["early_stopped"] is True
    assert state["iterations"] == len(validation) < 500
    assert state["best_epoch"] == int(np.argmin(validation)) + 1
    assert any(notice.code == "early_stopping" for notice in model.notices)


def test_without_early_stopping_every_epoch_is_trained() -> None:
    X, y = moons(n=100)

    model = make("mlp", early_stopping=False, max_epochs=15).fit(X, y)

    assert model.state()["iterations"] == 15
    assert model.state()["best_epoch"] is None
    assert [notice.code for notice in model.notices] == ["max_epochs"]


def test_each_snapshot_shows_what_each_layer_does() -> None:
    X, y = moons(n=200)
    context = RecordingTrainingContext()

    make("mlp", hidden_layers=2, units=8, max_epochs=20).fit(X, y, context)

    layers = context.snapshots[-1][1]["layers"]
    assert [layer["name"] for layer in layers] == ["Camada 1", "Camada 2", "Saída"]
    hidden = layers[0]
    assert len(hidden["activations"]) == len(hidden["stuck_units"]) == 8
    assert 0.0 <= hidden["stuck"] <= 1.0
    assert hidden["gradient"] > 0
    # As ativações medem-se nas linhas de treino, sem as 20 (10%) postas de lado para validação.
    assert sum(hidden["histogram"]["counts"]) == 180 * 8
    assert layers[-1]["activation"] == "softmax"
    assert "activations" not in layers[-1]


def test_a_relu_that_never_switches_on_is_counted_as_stuck() -> None:
    X, y = moons(n=100)
    model = MLPClassifier(MLPParams(hidden_layers=1, units=4, max_epochs=5)).fit(X, y)
    first = model._network[0]  # a primeira camada densa (o teste mexe-lhe por dentro)
    assert isinstance(first, torch.nn.Linear)

    with torch.no_grad():
        first.bias[:2] = -100.0  # dois neurónios com a soma sempre negativa: ReLU sempre a 0

    hidden = model.state()["layers"][0]
    assert hidden["stuck_units"][:2] == [True, True]
    assert hidden["stuck"] >= 0.5


def test_too_few_rows_for_validation_are_explained() -> None:
    X, y = moons(n=4)  # 10% de 4 linhas arredonda para 0: não sobra nenhuma para validar

    model = make("mlp", max_epochs=5).fit(X, y)

    assert "no_validation" in [notice.code for notice in model.notices]


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_every_activation_trains_and_is_named(activation: str) -> None:
    X, y = moons(n=100)

    state = make("mlp", activation=activation, max_epochs=10).fit(X, y).state()

    assert (
        state["activation"] == {"relu": "ReLU", "tanh": "tanh", "sigmoid": "sigmoide"}[activation]
    )
    assert state["architecture"] == "2 → 16 → 16 → 2"
