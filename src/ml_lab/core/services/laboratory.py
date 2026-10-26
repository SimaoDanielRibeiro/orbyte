"""Serviço do Laboratório: demonstrações de conceitos com um só controlo (ecrã Laboratório).

Cada demonstração treina modelos do catálogo em dados pequenos, feitos para mostrar uma
ideia, e devolve os números dos gráficos:

- **overfitting**: uma regressão polinomial num seno com ruído; o grau sobe e o erro de
  treino desce sempre, mas o de teste desce e depois volta a subir;
- **taxa de aprendizagem**: o gradiente descendente da regressão linear numa "taça" com
  dois pesos; passos pequenos arrastam-se, grandes fazem zigue-zague, enormes explodem;
- **o k do KNN**: a fronteira nas luas, de ilhas à volta de cada ponto (k = 1) até uma
  fronteira lisa demais (k grande);
- **regularização**: os pesos da Ridge e do Lasso a encolher à medida que o alpha cresce;
  no Lasso, os inúteis chegam a zero.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from ml_lab.config import Settings
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.evaluation.metrics.base import MetricInputs
from ml_lab.core.evaluation.metrics.regression import r2 as r2_metric
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.training.context import RecordingTrainingContext
from ml_lab.core.visualization_data.animation import TrainingAnimation, build_animation
from ml_lab.core.visualization_data.grid import probe_grid
from ml_lab.formatting import format_number, format_percent

DEGREES: Final = tuple(range(1, 16))
KS: Final = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100)
COMPARED_RATES: Final = (0.01, 0.1, 0.5)
EPOCHS: Final = 60
# Luas escolhidas para a curva clássica: k pequeno decora, k grande alisa demais, e o
# melhor fica no meio (perto de 15).
MOONS: Final = SourceSpec(
    "synthetic", {"generator": "moons", "seed": 2, "parameters": {"n_samples": 500, "noise": 0.35}}
)
MOONS_SEED: Final = 2


class LaboratoryDemo(StrEnum):
    """Uma demonstração do Laboratório (o nome é o do método e o da rota)."""

    OVERFITTING = "overfitting"
    LEARNING_RATE = "learning_rate"
    KNN = "knn"
    REGULARIZATION = "regularization"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _DEMO_LABELS[self]

    @property
    def models(self) -> frozenset[str]:
        """Os modelos cujo funcionamento a demonstração mostra (a página deles liga-se a ela)."""
        return _DEMO_MODELS[self]

    @property
    def topic(self) -> str:
        """O tema do painel "Porquê?" que explica a demonstração."""
        return _DEMO_TOPICS[self]


_DEMO_LABELS: Final[Mapping[LaboratoryDemo, str]] = MappingProxyType(
    {
        LaboratoryDemo.OVERFITTING: "Complexidade e overfitting",
        LaboratoryDemo.LEARNING_RATE: "Taxa de aprendizagem",
        LaboratoryDemo.KNN: "O k do KNN",
        LaboratoryDemo.REGULARIZATION: "Regularização",
    }
)

_DEMO_MODELS: Final[Mapping[LaboratoryDemo, frozenset[str]]] = MappingProxyType(
    {
        # A demonstração muda o grau desta regressão.
        LaboratoryDemo.OVERFITTING: frozenset({"polynomial_regression"}),
        # O mesmo passo do gradiente descendente, numa taça com dois pesos.
        LaboratoryDemo.LEARNING_RATE: frozenset(
            {"linear_regression_gd", "logistic_regression_gd", "mlp", "mlp_regression"}
        ),
        LaboratoryDemo.KNN: frozenset({"knn", "knn_numpy"}),
        LaboratoryDemo.REGULARIZATION: frozenset({"ridge", "lasso"}),
    }
)

_DEMO_TOPICS: Final[Mapping[LaboratoryDemo, str]] = MappingProxyType(
    {
        LaboratoryDemo.OVERFITTING: "bias_variance",
        LaboratoryDemo.LEARNING_RATE: "learning_rate_demo",
        LaboratoryDemo.KNN: "knn_k",
        LaboratoryDemo.REGULARIZATION: "regularization",
    }
)


def demos_for(model: str) -> tuple[LaboratoryDemo, ...]:
    """As demonstrações do Laboratório que mostram como ``model`` funciona."""
    return tuple(demo for demo in LaboratoryDemo if model in demo.models)


@dataclass(frozen=True, slots=True)
class Points:
    """Pontos (ou uma curva) no plano."""

    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OverfittingDemo:
    """O erro no treino e no teste para cada grau, e a curva do grau escolhido."""

    degree: int
    degrees: tuple[int, ...]
    train_errors: tuple[float, ...]
    test_errors: tuple[float, ...]
    curve: Points
    truth: Points
    train: Points
    test: Points
    notices: tuple[Notice, ...]


@dataclass(frozen=True, slots=True)
class LossCurve:
    """A loss época a época com uma taxa de aprendizagem."""

    rate: float
    losses: tuple[float, ...]
    diverged: bool


@dataclass(frozen=True, slots=True)
class LearningRateDemo:
    """As curvas da loss de várias taxas e o caminho dos pesos na superfície do erro."""

    rate: float
    curves: tuple[LossCurve, ...]
    path: Points
    surface_x: tuple[float, ...]
    surface_y: tuple[float, ...]
    surface: tuple[tuple[float, ...], ...]
    optimum: tuple[float, float]
    notices: tuple[Notice, ...]


@dataclass(frozen=True, slots=True)
class KnnDemo:
    """A exatidão para cada k e a fronteira do k escolhido."""

    k: int
    ks: tuple[int, ...]
    train_accuracy: tuple[float, ...]
    test_accuracy: tuple[float, ...]
    boundary: TrainingAnimation | None
    notices: tuple[Notice, ...]


@dataclass(frozen=True, slots=True)
class RegularizationDemo:
    """Os pesos de cada feature para cada alpha, e o R² no treino e no teste."""

    kind: Literal["ridge", "lasso"]
    alpha: float
    alphas: tuple[float, ...]
    feature_names: tuple[str, ...]
    paths: tuple[tuple[float, ...], ...]
    true_coefficients: tuple[float, ...]
    train_r2: tuple[float, ...]
    test_r2: tuple[float, ...]
    chosen: tuple[float, ...]
    notices: tuple[Notice, ...]


class LaboratoryService:
    """As demonstrações do ecrã Laboratório.

    Args:
        settings: As definições (o laboratório usa um serviço de datasets só seu, para os
            dados das demonstrações não aparecerem na lista do ecrã Dados).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._datasets = DatasetService(settings)

    def overfitting(self, degree: int) -> OverfittingDemo:
        """Uma regressão polinomial de grau ``degree`` num seno com ruído (20 pontos de treino)."""
        rng = np.random.default_rng(7)
        x_train, x_test = np.sort(rng.uniform(-3, 3, 20)), rng.uniform(-3, 3, 200)
        y_train = np.sin(x_train) + rng.normal(scale=0.3, size=20)
        y_test = np.sin(x_test) + rng.normal(scale=0.3, size=200)
        grid = np.linspace(-3, 3, 200)
        train_errors, test_errors = [], []
        curve = np.zeros(0)
        for chosen in DEGREES:
            model = _model("polynomial_regression", degree=chosen, alpha=1e-6)
            model.fit(x_train[:, None], y_train)
            train_errors.append(_rmse(y_train, model.predict(x_train[:, None])))
            test_errors.append(_rmse(y_test, model.predict(x_test[:, None])))
            if chosen == degree:
                curve = np.asarray(model.predict(grid[:, None]), dtype=np.float64)
        best = DEGREES[int(np.argmin(test_errors))]
        return OverfittingDemo(
            degree=degree,
            degrees=DEGREES,
            train_errors=_floats(train_errors),
            test_errors=_floats(test_errors),
            curve=Points(_floats(grid), _floats(curve)),
            truth=Points(_floats(grid), _floats(np.sin(grid))),
            train=Points(_floats(x_train), _floats(y_train)),
            test=Points(_floats(x_test), _floats(y_test)),
            notices=_read_overfitting(degree, best, train_errors, test_errors),
        )

    def learning_rate(self, rate: float) -> LearningRateDemo:
        """O gradiente descendente da regressão linear com dois pesos e a taxa ``rate``."""
        rng = np.random.default_rng(5)
        first = rng.normal(size=100)
        second = 0.8 * first + 0.6 * rng.normal(size=100)  # correlacionadas: a taça é alongada
        X = np.column_stack([first, second])
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = X @ np.array([2.0, -1.0]) + rng.normal(scale=0.5, size=100)
        y -= y.mean()
        curves = []
        path = Points((), ())
        for compared in sorted({*COMPARED_RATES, rate}):
            context = RecordingTrainingContext()
            model = _model(
                "linear_regression_gd",
                learning_rate=compared,
                max_epochs=EPOCHS,
                tol=1e-10,
                fit_intercept=False,
            )
            model.fit(X, y, context)
            losses = [loss for _, loss, _ in context.steps if loss is not None]
            diverged = any(notice.code == "diverged" for notice in model.notices)
            curves.append(LossCurve(compared, _floats(losses), diverged))
            if compared == rate:
                weights = [(0.0, 0.0)] + [
                    tuple(state["coefficients"]) for _, state in context.snapshots
                ]
                path = Points(_floats([w[0] for w in weights]), _floats([w[1] for w in weights]))
        optimum = np.linalg.lstsq(X, y, rcond=None)[0]
        axis_x = np.linspace(optimum[0] - 4, optimum[0] + 4, 60)
        axis_y = np.linspace(optimum[1] - 4, optimum[1] + 4, 60)
        grid_x, grid_y = np.meshgrid(axis_x, axis_y)
        weights_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        surface = np.mean((X @ weights_grid.T - y[:, None]) ** 2, axis=0).reshape(grid_x.shape)
        chosen = next(curve for curve in curves if curve.rate == rate)
        return LearningRateDemo(
            rate=rate,
            curves=tuple(curves),
            path=path,
            surface_x=_floats(axis_x),
            surface_y=_floats(axis_y),
            surface=tuple(_floats(row) for row in surface),
            optimum=(float(optimum[0]), float(optimum[1])),
            notices=_read_learning_rate(chosen, float(np.mean((X @ optimum - y) ** 2))),
        )

    def knn(self, k: int) -> KnnDemo:
        """O KNN (em NumPy) nas luas com ``k`` vizinhos, e a exatidão de outros k."""
        preparation = PreparationConfig(target="target")
        dataset = self._datasets.prepared(MOONS, preparation, MOONS_SEED).dataset
        train, test = dataset.train, dataset.test
        assert train.y is not None  # as luas têm alvo
        assert test.y is not None
        ks = tuple(value for value in sorted({*KS, k}) if value <= len(train))
        train_accuracy, test_accuracy = [], []
        boundary = None
        grid = probe_grid(dataset)
        for value in ks:
            model = _model("knn_numpy", n_neighbors=value).fit(train.X, train.y)
            train_accuracy.append(float(np.mean(model.predict(train.X) == train.y)))
            test_accuracy.append(float(np.mean(model.predict(test.X) == test.y)))
            if value == k and grid is not None:
                probe = model.predict_proba(grid.points)
                boundary = build_animation(
                    dataset, grid, [(0, {"probe": probe})], classes=model.classes()
                )
        return KnnDemo(
            k=k,
            ks=ks,
            train_accuracy=_floats(train_accuracy),
            test_accuracy=_floats(test_accuracy),
            boundary=boundary,
            notices=_read_knn(k, ks, train_accuracy, test_accuracy),
        )

    def regularization(self, kind: Literal["ridge", "lasso"], alpha: float) -> RegularizationDemo:
        """Os pesos da Ridge ou do Lasso para uma gama de alphas (só 3 das 8 features contam)."""
        rng = np.random.default_rng(11)
        truth = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        X_train, X_test = rng.normal(size=(40, 8)), rng.normal(size=(200, 8))
        y_train = X_train @ truth + rng.normal(scale=2.0, size=40)
        y_test = X_test @ truth + rng.normal(scale=2.0, size=200)
        low, high = (-3, 2) if kind == "lasso" else (-2, 4)
        alphas = np.logspace(low, high, 30)
        paths, train_r2, test_r2 = [], [], []
        for value in alphas:
            model = self._regularized(kind, float(value)).fit(X_train, y_train)
            paths.append(model.state()["coefficients"])
            train_r2.append(_r2(y_train, model.predict(X_train)))
            test_r2.append(_r2(y_test, model.predict(X_test)))
        chosen = self._regularized(kind, alpha).fit(X_train, y_train).state()["coefficients"]
        columns = np.asarray(paths).T
        return RegularizationDemo(
            kind=kind,
            alpha=alpha,
            alphas=_floats(alphas),
            feature_names=tuple(f"x{index + 1}" for index in range(8)),
            paths=tuple(_floats(column) for column in columns),
            true_coefficients=_floats(truth),
            train_r2=_floats(train_r2),
            test_r2=_floats(test_r2),
            chosen=_floats(chosen),
            notices=_read_regularization(kind, alpha, alphas, test_r2, chosen),
        )

    def _regularized(self, kind: Literal["ridge", "lasso"], alpha: float) -> LabModel[Any]:
        if kind == "lasso":
            return _model("lasso", alpha=alpha, max_iter=100_000)
        return _model("ridge", alpha=alpha)


def _model(name: str, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    return model_class(model_class.info.hyperparameters.parse(values, model=name))


def _floats(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _rmse(actual: NDArray[np.float64], predicted: NDArray[Any]) -> float:
    return float(np.sqrt(np.mean((actual - np.asarray(predicted, dtype=np.float64)) ** 2)))


def _r2(actual: NDArray[np.float64], predicted: NDArray[Any]) -> float:
    return r2_metric(MetricInputs(y_true=actual, y_pred=np.asarray(predicted, dtype=np.float64)))


def _read_overfitting(
    degree: int, best: int, train: list[float], test: list[float]
) -> tuple[Notice, ...]:
    chosen = DEGREES.index(degree)
    errors = (
        f"Com grau {degree}, o erro (RMSE) é {format_number(train[chosen], 3)} no treino e "
        f"{format_number(test[chosen], 3)} no teste; o menor erro de teste é com grau {best}."
    )
    if degree > best + 2 and test[chosen] > 1.2 * min(test):
        return (
            Notice.warning(
                "overfitting",
                f"{errors} A curva passa perto de todos os pontos de treino, mas ondula entre "
                "eles: decorou o ruído (overfitting).",
            ),
        )
    if degree < best and test[chosen] > 1.2 * min(test):
        return (
            Notice.info(
                "underfitting",
                f"{errors} Uma curva tão simples não consegue seguir o seno: o erro é alto no "
                "treino e no teste (underfitting).",
            ),
        )
    return (Notice.info("good_fit", f"{errors} Este grau segue a forma do seno sem o ruído."),)


def _read_learning_rate(chosen: LossCurve, lowest: float) -> tuple[Notice, ...]:
    """A leitura da curva da taxa escolhida, comparada com a loss no fundo da taça."""
    rate = format_number(chosen.rate, 3)
    losses = chosen.losses
    if chosen.diverged:
        return (
            Notice.warning(
                "diverged",
                f"Com taxa {rate}, cada passo salta por cima do fundo da taça e cai mais alto do "
                "outro lado: a loss explode, e o treino pára.",
            ),
        )
    if len(losses) >= 2 and losses[-1] > losses[0]:
        return (
            Notice.warning(
                "growing",
                f"Com taxa {rate}, a loss está a subir: os passos são grandes demais e o caminho "
                "faz um zigue-zague cada vez mais largo. Com mais épocas, explodia.",
            ),
        )
    if losses and losses[-1] > 1.05 * lowest:
        return (
            Notice.info(
                "slow",
                f"Com taxa {rate}, os passos são pequenos: ao fim de {len(losses)} épocas o "
                "caminho ainda não chegou ao fundo da taça (a cruz).",
            ),
        )
    return (
        Notice.info(
            "converging",
            f"Com taxa {rate}, o caminho desce a taça até ao fundo (a cruz). Repara que não vai "
            "a direito: em cada ponto segue a direção mais inclinada.",
        ),
    )


def _read_knn(
    k: int, ks: tuple[int, ...], train: list[float], test: list[float]
) -> tuple[Notice, ...]:
    chosen = ks.index(k)
    best = ks[int(np.argmax(test))]
    accuracies = (
        f"Com k = {k}, a exatidão é {format_percent(train[chosen])} no treino e "
        f"{format_percent(test[chosen])} no teste; o melhor teste é com k = {best}."
    )
    if k < best / 2:
        memorised = " e o treino quase perfeito" if train[chosen] >= 0.98 else ""
        return (
            Notice.warning(
                "k_small",
                f"{accuracies} Com poucos vizinhos, a fronteira segue o ruído: ilhas à volta "
                f"dos pontos isolados{memorised} (overfitting).",
            ),
        )
    if k > 3 * best:
        return (
            Notice.info(
                "k_large",
                f"{accuracies} Com tantos vizinhos, a fronteira fica lisa demais e apaga a "
                "forma das luas (underfitting).",
            ),
        )
    return (Notice.info("k_fine", f"{accuracies} A fronteira segue as luas sem seguir o ruído."),)


def _read_regularization(
    kind: str, alpha: float, alphas: NDArray[np.float64], test_r2: list[float], chosen: Any
) -> tuple[Notice, ...]:
    best = float(alphas[int(np.argmax(test_r2))])
    nearest = int(np.argmin(np.abs(np.log10(alphas) - math.log10(alpha))))
    message = (
        f"Com alpha = {format_number(alpha, 3)}, o R² no teste é "
        f"{format_number(test_r2[nearest], 3)}; o melhor alpha nestes dados anda por "
        f"{format_number(best, 3)}."
    )
    if kind == "lasso":
        zeros = int(np.sum(np.abs(np.asarray(chosen)) < 1e-10))
        message += f" {zeros} dos 8 pesos ficaram exatamente a zero (só 3 features contam)."
    return (Notice.info("regularization", message),)
