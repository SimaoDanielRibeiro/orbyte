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
  no Lasso, os inúteis chegam a zero;
- **limiar de decisão**: uma regressão logística em dados desequilibrados (1 positivo em
  10); o limiar troca alarmes falsos por casos perdidos, e a exatidão quase não se mexe;
- **quantidade de dados**: uma árvore sem limite de profundidade e uma regressão logística
  com cada vez mais linhas; o modelo simples ganha com poucas, o flexível com muitas.

Os valores que cada controlo oferece também se decidem aqui (``demos``): são uma escolha
didática (de onde a taxa de aprendizagem se arrasta até onde explode), e a interface desenha
os controlos a partir deles. Cada cartão traz também o guia da demonstração (a ideia, o que
experimentar e o que fica), para o ecrã explicar sem escrever texto seu (ADR 0051).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from ml_lab.config import Settings
from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitConfig
from ml_lab.core.evaluation.metrics.base import MetricInputs
from ml_lab.core.evaluation.metrics.regression import r2 as r2_metric
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.training.context import RecordingTrainingContext
from ml_lab.core.visualization_data.animation import (
    PointCloud,
    TrainingAnimation,
    build_animation,
)
from ml_lab.core.visualization_data.grid import ProbeGrid, probe_grid
from ml_lab.formatting import format_number, format_percent, format_value

DEGREES: Final = tuple(range(1, 16))
KS: Final = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100)
RATES: Final = (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5)
"""As taxas que o controlo oferece: das que se arrastam às que explodem (acima de 1)."""
COMPARED_RATES: Final = (0.01, 0.1, 0.5)
EPOCHS: Final = 60
DEFAULT_DEGREE: Final = 3
DEFAULT_RATE: Final = 0.1
DEFAULT_K: Final = 5
ALPHA_DECADES: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {"ridge": (-2, 4), "lasso": (-3, 2)}
)
"""As potências de 10 entre as quais anda o alpha de cada modelo (o Lasso apaga pesos antes)."""
# Luas escolhidas para a curva clássica: k pequeno decora, k grande alisa demais, e o
# melhor fica no meio (perto de 15).
MOONS: Final = SourceSpec(
    "synthetic", {"generator": "moons", "seed": 2, "parameters": {"n_samples": 500, "noise": 0.35}}
)
MOONS_SEED: Final = 2
THRESHOLDS: Final = tuple(round(0.05 * step, 2) for step in range(1, 20))
"""Os limiares que o controlo oferece, de 5% a 95%."""
DEFAULT_THRESHOLD: Final = 0.5
POSITIVE_SHARE: Final = 0.1
"""A parte de positivos nos dados do limiar: poucos, como as fraudes ou as doenças raras."""
SIZES: Final = (10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750)
"""As quantidades de linhas de treino que a curva compara (750 são todas)."""
DEFAULT_SIZE: Final = 10
"""A demonstração abre com poucas linhas: o guia lê-se da esquerda para a direita."""
REPEATS: Final = 5
"""Amostras diferentes por quantidade: com poucas linhas, uma só amostra dependia da sorte."""
# Mais luas, e menos ruído do que as do KNN: com 750 linhas a árvore já desenha as luas, e a
# regressão logística (uma reta) fica-se pelos 85%. O teste (500 linhas) é sempre o mesmo.
MANY_MOONS: Final = SourceSpec(
    "synthetic",
    {"generator": "moons", "seed": 4, "parameters": {"n_samples": 1250, "noise": 0.25}},
)
MANY_MOONS_SEED: Final = 4


class LaboratoryDemo(StrEnum):
    """Uma demonstração do Laboratório (o nome é o do método e o da rota)."""

    OVERFITTING = "overfitting"
    LEARNING_RATE = "learning_rate"
    KNN = "knn"
    REGULARIZATION = "regularization"
    THRESHOLD = "threshold"
    DATA_SIZE = "data_size"

    @property
    def label(self) -> str:
        """Nome a mostrar, em inglês."""
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
        LaboratoryDemo.OVERFITTING: "Overfitting",
        LaboratoryDemo.LEARNING_RATE: "Learning rate",
        LaboratoryDemo.KNN: "The k in KNN",
        LaboratoryDemo.REGULARIZATION: "Regularisation",
        LaboratoryDemo.THRESHOLD: "Decision threshold",
        LaboratoryDemo.DATA_SIZE: "Data size",
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
        # O limiar decide com as probabilidades destes dois (a demonstração usa o primeiro).
        LaboratoryDemo.THRESHOLD: frozenset({"logistic_regression", "logistic_regression_gd"}),
        # A árvore sem limite é o modelo flexível que precisa de muitos dados.
        LaboratoryDemo.DATA_SIZE: frozenset({"decision_tree"}),
    }
)

_DEMO_TOPICS: Final[Mapping[LaboratoryDemo, str]] = MappingProxyType(
    {
        LaboratoryDemo.OVERFITTING: "bias_variance",
        LaboratoryDemo.LEARNING_RATE: "learning_rate_demo",
        LaboratoryDemo.KNN: "knn_k",
        LaboratoryDemo.REGULARIZATION: "regularization",
        LaboratoryDemo.THRESHOLD: "decision_threshold",
        LaboratoryDemo.DATA_SIZE: "learning_curve",
    }
)


_DEMO_SUMMARIES: Final[Mapping[LaboratoryDemo, str]] = MappingProxyType(
    {
        LaboratoryDemo.OVERFITTING: (
            "A polynomial regression on 20 points of a noisy sine. Raise the degree and watch the "
            "training error keep falling, while the test error falls and then rises again."
        ),
        LaboratoryDemo.LEARNING_RATE: (
            "Linear regression with two weights, trained by gradient descent for "
            f"{EPOCHS} epochs. The error is a bowl (the contour lines) and the path shows the "
            "weights epoch by epoch."
        ),
        LaboratoryDemo.KNN: (
            "KNN (in NumPy) on the noisy moons. With few neighbours, the boundary follows every "
            "point; with many, it gets too smooth. The best k is in between."
        ),
        LaboratoryDemo.REGULARIZATION: (
            "A regression with 8 features where only 3 count (x1, x2 and x3) and only 40 training "
            "rows. Watch the weights shrink as alpha grows."
        ),
        LaboratoryDemo.THRESHOLD: (
            "A logistic regression on imbalanced data, where only 1 row in 10 is positive. Move "
            "the threshold that turns the predicted probability into a decision, and watch "
            "precision, recall and the confusion matrix change (all on the test data)."
        ),
        LaboratoryDemo.DATA_SIZE: (
            f"A decision tree with no depth limit and a logistic regression, trained on more and "
            f"more rows of the noisy moons and always tested on the same 500. Each amount is the "
            f"mean of {REPEATS} different samples, and the band shows how much they vary."
        ),
    }
)


@dataclass(frozen=True, slots=True)
class DemoGuide:
    """O guia de uma demonstração: a ideia, o que experimentar e o que fica.

    Attributes:
        idea: O conceito, em duas ou três frases, antes de mexer em nada.
        steps: O que experimentar, por ordem: cada passo diz que valor escolher e o que ver.
        takeaway: O que fica, numa frase, para levar para os modelos a sério.
    """

    idea: str
    steps: tuple[str, ...]
    takeaway: str


_DEMO_GUIDES: Final[Mapping[LaboratoryDemo, DemoGuide]] = MappingProxyType(
    {
        LaboratoryDemo.OVERFITTING: DemoGuide(
            idea=(
                "A model can fit its training data too well. A flexible model also learns the "
                "noise, and the noise doesn't repeat in new data: that's why a model is always "
                "judged on data it hasn't seen."
            ),
            steps=(
                "Start at degree 1: a straight line can't follow the sine, so the error is high "
                "on both the training and the test data (underfitting).",
                "Raise the degree to 3, 4 or 5: the curve follows the sine, and the test error "
                "reaches its lowest.",
                "Go to degree 12 or more: the curve passes through the training points but "
                "wiggles between them, and the test error climbs (overfitting).",
            ),
            takeaway=(
                "The training error always falls as the model gets more flexible; only the test "
                "error tells you when to stop."
            ),
        ),
        LaboratoryDemo.LEARNING_RATE: DemoGuide(
            idea=(
                "Gradient descent finds the weights by walking downhill on the error, one step "
                "per epoch. The learning rate is the size of each step."
            ),
            steps=(
                f"Pick 0.01: the steps are tiny, and after {EPOCHS} epochs the path is still far "
                "from the bottom of the bowl.",
                "Try 0.1 to 0.3: the path reaches the bottom quickly.",
                "Push it above 1: each step jumps over the valley, the path zigzags and the loss "
                "blows up.",
            ),
            takeaway=(
                "Too small wastes time; too big never settles. A good loss curve falls quickly "
                "and then flattens."
            ),
        ),
        LaboratoryDemo.KNN: DemoGuide(
            idea=(
                "KNN classifies a point by the vote of its k nearest training points. The k "
                "decides how local that vote is."
            ),
            steps=(
                "Set k = 1: every training point gets its own island, and the training accuracy "
                "is 100%.",
                "Move k to 10 or 20: the islands disappear, and the boundary follows the two "
                "moons.",
                "Push k to 75 or 100: the vote takes in points from the other moon, and the "
                "boundary gets too smooth.",
            ),
            takeaway=(
                "A small k follows the noise (variance); a large k misses the shape (bias). "
                "Choose k by the test accuracy, or by cross-validation, never by the training "
                "accuracy."
            ),
        ),
        LaboratoryDemo.REGULARIZATION: DemoGuide(
            idea=(
                "Regularisation adds the size of the weights to the error the model minimises. "
                "The model then prefers small weights, and ends up simpler."
            ),
            steps=(
                "With a small alpha, the five useless weights are far from zero: with only 40 "
                "rows, the model fits the noise.",
                "Raise alpha: the useless weights shrink first, and the test R² improves.",
                "Switch between Lasso and Ridge: in Lasso the useless weights reach exactly "
                "zero, so it also picks the features.",
                "Go to the largest alpha: the useful weights shrink too, and the R² falls on "
                "both sets.",
            ),
            takeaway=(
                "Ridge shrinks every weight; Lasso can drop features entirely. Too much "
                "regularisation wipes out what matters too."
            ),
        ),
        LaboratoryDemo.THRESHOLD: DemoGuide(
            idea=(
                "A classifier gives each row a probability, and the decision compares it with a "
                "threshold (50% by default). Moving the threshold trades false alarms for "
                "missed cases."
            ),
            steps=(
                "Lower the threshold to 10%: the model flags many rows and finds almost every "
                "positive (high recall), but many flags are false alarms (low precision).",
                "Raise it to 90%: the model only flags the rows it's sure of. Almost every flag "
                "is right, but it misses most of the positives.",
                "Watch the accuracy as you move it: it barely changes, because answering "
                "“negative” for everyone is already right 9 times in 10.",
            ),
            takeaway=(
                "The threshold is a decision about costs, not part of the model: a missed "
                "disease and a false alarm don't cost the same. With imbalanced classes, look "
                "at precision, recall and F1, not accuracy."
            ),
        ),
        LaboratoryDemo.DATA_SIZE: DemoGuide(
            idea=(
                "What a model can learn depends on how many examples it sees. A flexible model "
                "needs many before its rules stop following chance; a simple one learns quickly "
                "but stops improving early."
            ),
            steps=(
                "Start with 10 rows: the tree gets every training row right, but the simpler "
                "logistic regression does better on the test.",
                "Go to 30, 50 or 100 rows: the tree overtakes the logistic regression, and its "
                "boundary starts to look like the moons.",
                "Go to 750 rows: the tree keeps improving and its band narrows, while the "
                "logistic regression stopped long ago (a straight line can't follow the moons).",
            ),
            takeaway=(
                "More data reduces variance, not bias: when a simple model stops improving, it "
                "needs more flexibility, not more rows."
            ),
        ),
    }
)


def demos_for(model: str) -> tuple[LaboratoryDemo, ...]:
    """As demonstrações do Laboratório que mostram como ``model`` funciona."""
    return tuple(demo for demo in LaboratoryDemo if model in demo.models)


@dataclass(frozen=True, slots=True)
class DemoControl:
    """O controlo de uma demonstração: os valores que se podem escolher e o de partida.

    Attributes:
        label: O nome a mostrar.
        values: Os valores que o controlo oferece, por ordem crescente.
        default: O valor com que a demonstração abre (um dos ``values``).
    """

    label: str
    values: tuple[float, ...]
    default: float


@dataclass(frozen=True, slots=True)
class DemoVariant:
    """Uma versão de uma demonstração, com o seu controlo (a Ridge e o Lasso).

    Attributes:
        name: O valor que a rota da demonstração recebe (o ``kind`` da regularização).
        label: O nome a mostrar.
        control: O controlo desta versão (o alpha anda numa gama diferente em cada uma).
    """

    name: str
    label: str
    control: DemoControl


@dataclass(frozen=True, slots=True)
class DemoCard:
    """Uma demonstração do Laboratório, pronta a mostrar.

    Attributes:
        name: A demonstração (o nome do método e o da rota).
        summary: O que se vê e o que se pode mexer.
        topic: O tema do painel "Porquê?" que a explica.
        guide: A ideia, o que experimentar e o que fica.
        control: O controlo, nas demonstrações com uma só versão.
        variants: As versões, cada uma com o seu controlo (só a regularização as tem).
        variant_label: O nome a mostrar da escolha da versão.
        default_variant: A versão com que a demonstração abre.
    """

    name: LaboratoryDemo
    summary: str
    topic: str
    guide: DemoGuide
    control: DemoControl | None
    variants: tuple[DemoVariant, ...] = ()
    variant_label: str | None = None
    default_variant: str | None = None


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


@dataclass(frozen=True, slots=True)
class ThresholdDemo:
    """As métricas para cada limiar, a matriz de confusão do escolhido e as probabilidades.

    Attributes:
        threshold: O limiar escolhido.
        thresholds: Os limiares comparados (com o escolhido).
        precision: Dos que o modelo marca como positivos, a parte que o é (``None`` quando não
            marca nenhum: não há de onde tirar a parte).
        recall: Dos positivos, a parte que o modelo encontra.
        f1: A média harmónica das duas.
        accuracy: A parte das linhas certas.
        baseline_accuracy: A exatidão de responder sempre "negativo" (a classe maior).
        positive_share: A parte de positivos no teste.
        class_labels: O nome de cada classe (negativa e positiva).
        confusion: A matriz de confusão do limiar escolhido; as linhas são a classe real e as
            colunas a prevista, pela ordem de ``class_labels``.
        negatives: A probabilidade de "positivo" que o modelo dá a cada linha negativa do teste.
        positives: A mesma probabilidade, para cada linha positiva do teste.
        notices: A leitura do limiar escolhido.
    """

    threshold: float
    thresholds: tuple[float, ...]
    precision: tuple[float | None, ...]
    recall: tuple[float, ...]
    f1: tuple[float, ...]
    accuracy: tuple[float, ...]
    baseline_accuracy: float
    positive_share: float
    class_labels: tuple[str, str]
    confusion: tuple[tuple[int, ...], ...]
    negatives: tuple[float, ...]
    positives: tuple[float, ...]
    notices: tuple[Notice, ...]


@dataclass(frozen=True, slots=True)
class DataSizeDemo:
    """A exatidão de uma árvore e de uma regressão logística para cada quantidade de dados.

    Cada valor é a média das amostras (``REPEATS``); o ``spread`` é o desvio padrão entre elas.

    Attributes:
        size: A quantidade de linhas de treino escolhida.
        sizes: As quantidades comparadas (com a escolhida).
        tree_train: A exatidão da árvore nas suas linhas de treino.
        tree_test: A exatidão da árvore no teste.
        tree_spread: O desvio da exatidão da árvore no teste, entre as amostras.
        logistic_train: A exatidão da regressão logística nas suas linhas de treino.
        logistic_test: A exatidão da regressão logística no teste.
        logistic_spread: O desvio da exatidão da regressão logística no teste.
        test_rows: As linhas do teste (sempre as mesmas).
        boundary: A fronteira da árvore treinada com a quantidade escolhida (a primeira amostra),
            com as linhas que a treinaram.
        notices: A leitura da quantidade escolhida.
    """

    size: int
    sizes: tuple[int, ...]
    tree_train: tuple[float, ...]
    tree_test: tuple[float, ...]
    tree_spread: tuple[float, ...]
    logistic_train: tuple[float, ...]
    logistic_test: tuple[float, ...]
    logistic_spread: tuple[float, ...]
    test_rows: int
    boundary: TrainingAnimation | None
    notices: tuple[Notice, ...]


class LaboratoryService:
    """As demonstrações do ecrã Laboratório.

    Args:
        settings: As definições (o laboratório usa um serviço de datasets só seu, para os
            dados das demonstrações não aparecerem na lista do ecrã Dados).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._datasets = DatasetService(settings)

    def demos(self) -> tuple[DemoCard, ...]:
        """As demonstrações, com o que mostram e os valores que o controlo de cada uma oferece."""
        alphas = {kind: _alpha_values(kind) for kind in ALPHA_DECADES}
        return (
            _card(
                LaboratoryDemo.OVERFITTING,
                DemoControl("Polynomial degree", _floats(DEGREES), DEFAULT_DEGREE),
            ),
            _card(LaboratoryDemo.LEARNING_RATE, DemoControl("Learning rate", RATES, DEFAULT_RATE)),
            _card(
                LaboratoryDemo.KNN,
                DemoControl("Number of neighbours (k)", _floats(KS), DEFAULT_K),
            ),
            replace(
                _card(LaboratoryDemo.REGULARIZATION, None),
                variants=tuple(
                    DemoVariant(
                        kind,
                        label,
                        # O alpha começa a meio da gama: nem sem efeito, nem a apagar tudo.
                        DemoControl(
                            "Alpha (strength of the regularisation)",
                            alphas[kind],
                            alphas[kind][len(alphas[kind]) // 2],
                        ),
                    )
                    for kind, label in (("ridge", "Ridge (L2)"), ("lasso", "Lasso (L1)"))
                ),
                variant_label="Model",
                # O Lasso mostra melhor a ideia: os pesos inúteis chegam exatamente a zero.
                default_variant="lasso",
            ),
            _card(
                LaboratoryDemo.THRESHOLD,
                DemoControl("Decision threshold", THRESHOLDS, DEFAULT_THRESHOLD),
            ),
            _card(
                LaboratoryDemo.DATA_SIZE,
                DemoControl("Training rows", _floats(SIZES), DEFAULT_SIZE),
            ),
        )

    def overfitting(self, degree: int) -> OverfittingDemo:
        """Uma regressão polinomial de grau ``degree`` num seno com ruído (20 pontos de treino).

        Raises:
            InvalidParameterError: Se o grau não for um dos que a demonstração compara.
        """
        if degree not in DEGREES:
            raise InvalidParameterError(
                f"The polynomial degree must be between {DEGREES[0]} and {DEGREES[-1]}."
            )
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
        """O gradiente descendente da regressão linear com dois pesos e a taxa ``rate``.

        Raises:
            InvalidParameterError: Se a taxa não for positiva.
        """
        if not (math.isfinite(rate) and rate > 0):
            raise InvalidParameterError("The learning rate must be a positive number.")
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
        """O KNN (em NumPy) nas luas com ``k`` vizinhos, e a exatidão de outros k.

        Raises:
            InvalidParameterError: Se ``k`` não estiver entre 1 e o número de linhas de treino.
        """
        preparation = PreparationConfig(target="target")
        dataset = self._datasets.prepared(MOONS, preparation, MOONS_SEED).dataset
        train, test = dataset.train, dataset.test
        assert train.y is not None  # as luas têm alvo
        assert test.y is not None
        if not 1 <= k <= len(train):
            raise InvalidParameterError(
                f"The number of neighbours must be between 1 and {len(train)} (the training rows)."
            )
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
        """Os pesos da Ridge ou do Lasso para uma gama de alphas (só 3 das 8 features contam).

        Raises:
            InvalidParameterError: Se o alpha não for positivo.
        """
        if not (math.isfinite(alpha) and alpha > 0):
            raise InvalidParameterError("Alpha must be a positive number.")
        rng = np.random.default_rng(11)
        truth = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        X_train, X_test = rng.normal(size=(40, 8)), rng.normal(size=(200, 8))
        y_train = X_train @ truth + rng.normal(scale=2.0, size=40)
        y_test = X_test @ truth + rng.normal(scale=2.0, size=200)
        low, high = ALPHA_DECADES[kind]
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

    def threshold(self, threshold: float) -> ThresholdDemo:
        """Uma regressão logística em dados desequilibrados, a decidir com o limiar ``threshold``.

        Raises:
            InvalidParameterError: Se o limiar não estiver entre 0 e 1.
        """
        if not (math.isfinite(threshold) and 0 < threshold < 1):
            raise InvalidParameterError(
                "The threshold must be a probability between 0 and 1 (0.5, for example)."
            )
        rng = np.random.default_rng(13)
        # Os positivos (1 em 10) ficam deslocados mas não separados: há sempre linhas dos dois
        # lados de qualquer limiar, e cada escolha tem um preço.
        y = (rng.uniform(size=2000) < POSITIVE_SHARE).astype(np.int64)
        X = rng.normal(size=(2000, 2)) + np.where(y[:, None] == 1, 1.8, 0.0)
        model = _model("logistic_regression").fit(X[:1000], y[:1000])
        classes = model.classes()
        column = int(np.flatnonzero(classes == 1)[0]) if classes is not None else 1
        probability = np.asarray(model.predict_proba(X[1000:]), dtype=np.float64)[:, column]
        actual = y[1000:] == 1
        thresholds = tuple(sorted({*THRESHOLDS, threshold}))
        precision: list[float | None] = []
        recall, f1, accuracy = [], [], []
        for value in thresholds:
            tn, fp, fn, tp = _confusion(probability >= value, actual)
            precision.append(tp / (tp + fp) if tp + fp else None)
            recall.append(tp / (tp + fn))
            f1.append(2 * tp / (2 * tp + fp + fn))
            accuracy.append((tp + tn) / len(actual))
        tn, fp, fn, tp = _confusion(probability >= threshold, actual)
        share = float(np.mean(actual))
        chosen = thresholds.index(threshold)
        return ThresholdDemo(
            threshold=threshold,
            thresholds=thresholds,
            precision=tuple(precision),
            recall=_floats(recall),
            f1=_floats(f1),
            accuracy=_floats(accuracy),
            baseline_accuracy=1 - share,
            positive_share=share,
            class_labels=("Negative", "Positive"),
            confusion=((tn, fp), (fn, tp)),
            # Uma linha, um ponto: quatro casas chegam para a posição no desenho.
            negatives=tuple(round(float(value), 4) for value in probability[~actual]),
            positives=tuple(round(float(value), 4) for value in probability[actual]),
            notices=_read_threshold(
                threshold,
                precision[chosen],
                recall[chosen],
                accuracy[chosen],
                share,
                best=thresholds[int(np.argmax(f1))],
            ),
        )

    def data_size(self, size: int) -> DataSizeDemo:
        """Uma árvore sem limite e uma regressão logística treinadas com ``size`` linhas das luas.

        Raises:
            InvalidParameterError: Se a quantidade não estiver entre a menor da curva e as linhas
                de treino.
        """
        preparation = PreparationConfig(target="target", split=SplitConfig(test_size=0.4))
        dataset = self._datasets.prepared(MANY_MOONS, preparation, MANY_MOONS_SEED).dataset
        train, test = dataset.train, dataset.test
        assert train.y is not None  # as luas têm alvo
        assert test.y is not None
        if not SIZES[0] <= size <= len(train):
            raise InvalidParameterError(
                f"The training rows must be between {SIZES[0]} and {len(train)}."
            )
        sizes = tuple(sorted({*SIZES, size}))
        orders = _balanced_orders(train.y, REPEATS, np.random.default_rng(17))
        curves: dict[str, list[tuple[float, float, float]]] = {"tree": [], "logistic": []}
        for value in sizes:
            for name, values in (("tree", {"max_depth": 0}), ("logistic", {})):
                fits = []
                for order in orders:
                    rows = order[:value]
                    fitted = _model(_DATA_SIZE_MODELS[name], **values).fit(
                        train.X[rows], train.y[rows]
                    )
                    fits.append(
                        (
                            float(np.mean(fitted.predict(train.X[rows]) == train.y[rows])),
                            float(np.mean(fitted.predict(test.X) == test.y)),
                        )
                    )
                scores = np.asarray(fits)
                curves[name].append(
                    (float(scores[:, 0].mean()), float(scores[:, 1].mean()), scores[:, 1].std())
                )
        tree, logistic = np.asarray(curves["tree"]), np.asarray(curves["logistic"])
        grid = probe_grid(dataset)
        boundary = None
        if grid is not None:
            rows = orders[0][:size]
            fitted = _model("decision_tree", max_depth=0).fit(train.X[rows], train.y[rows])
            boundary = _boundary(dataset, grid, fitted, rows)
        return DataSizeDemo(
            size=size,
            sizes=sizes,
            tree_train=_floats(tree[:, 0]),
            tree_test=_floats(tree[:, 1]),
            tree_spread=_floats(tree[:, 2]),
            logistic_train=_floats(logistic[:, 0]),
            logistic_test=_floats(logistic[:, 1]),
            logistic_spread=_floats(logistic[:, 2]),
            test_rows=len(test),
            boundary=boundary,
            notices=_read_data_size(size, sizes, tree[:, 0], tree[:, 1], logistic[:, 1]),
        )

    def _regularized(self, kind: Literal["ridge", "lasso"], alpha: float) -> LabModel[Any]:
        if kind == "lasso":
            return _model("lasso", alpha=alpha, max_iter=100_000)
        return _model("ridge", alpha=alpha)


_DATA_SIZE_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {"tree": "decision_tree", "logistic": "logistic_regression"}
)


def _card(demo: LaboratoryDemo, control: DemoControl | None) -> DemoCard:
    return DemoCard(
        name=demo,
        summary=_DEMO_SUMMARIES[demo],
        topic=demo.topic,
        guide=_DEMO_GUIDES[demo],
        control=control,
    )


def _model(name: str, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    return model_class(model_class.info.hyperparameters.parse(values, model=name))


def _confusion(
    predicted: NDArray[np.bool_], actual: NDArray[np.bool_]
) -> tuple[int, int, int, int]:
    """A matriz de confusão de duas classes: ``(tn, fp, fn, tp)``."""
    return (
        int(np.sum(~predicted & ~actual)),
        int(np.sum(predicted & ~actual)),
        int(np.sum(~predicted & actual)),
        int(np.sum(predicted & actual)),
    )


def _balanced_orders(
    y: NDArray[Any], repeats: int, rng: np.random.Generator
) -> list[NDArray[np.int64]]:
    """Ordens das linhas de treino que alternam as classes.

    Assim, as primeiras linhas de qualquer ordem têm as duas classes, na mesma proporção: com
    10 linhas, uma amostra só de uma classe não deixava treinar a regressão logística.
    """
    orders = []
    for _ in range(repeats):
        groups = [rng.permutation(np.flatnonzero(y == label)) for label in np.unique(y)]
        longest = max(len(group) for group in groups)
        order = [group[index] for index in range(longest) for group in groups if index < len(group)]
        orders.append(np.asarray(order, dtype=np.int64))
    return orders


def _boundary(
    dataset: Dataset, grid: ProbeGrid, model: LabModel[Any], rows: NDArray[np.int64]
) -> TrainingAnimation | None:
    """A fronteira de ``model``, com só as linhas de treino que o treinaram por baixo dela.

    A grelha cobre todas as linhas de treino, para a vista não mudar de uma quantidade para a
    outra; os pontos desenhados são os da amostra.
    """
    animation = build_animation(
        dataset, grid, [(0, {"probe": model.predict_proba(grid.points)})], classes=model.classes()
    )
    projection, target = dataset.projection, dataset.train.y
    if animation is None or projection is None or target is None:
        return animation
    plane = projection.project(dataset.train.X[rows])
    labels = tuple(dataset.class_labels[int(code)] for code in target[rows])
    return replace(
        animation,
        points=PointCloud(x=_floats(plane[:, 0]), y=_floats(plane[:, 1]), labels=labels),
    )


def _floats(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _alpha_values(kind: str) -> tuple[float, ...]:
    """Alphas "redondos" (1; 2,5; 5 e 7,5 de cada potência de 10), na gama do modelo ``kind``."""
    low, high = ALPHA_DECADES[kind]
    values = (
        float(f"{step * 10.0**exponent:.6g}")
        for exponent in range(low, high + 1)
        for step in (1.0, 2.5, 5.0, 7.5)
    )
    return tuple(value for value in values if value <= 10.0**high)


def _rmse(actual: NDArray[np.float64], predicted: NDArray[Any]) -> float:
    return float(np.sqrt(np.mean((actual - np.asarray(predicted, dtype=np.float64)) ** 2)))


def _r2(actual: NDArray[np.float64], predicted: NDArray[Any]) -> float:
    return r2_metric(MetricInputs(y_true=actual, y_pred=np.asarray(predicted, dtype=np.float64)))


def _read_overfitting(
    degree: int, best: int, train: list[float], test: list[float]
) -> tuple[Notice, ...]:
    chosen = DEGREES.index(degree)
    errors = (
        f"With degree {degree}, the error (RMSE) is {format_number(train[chosen], 3)} on the "
        f"training data and {format_number(test[chosen], 3)} on the test data; the lowest test "
        f"error is with degree {best}."
    )
    if degree > best + 2 and test[chosen] > 1.2 * min(test):
        return (
            Notice.warning(
                "overfitting",
                f"{errors} The curve passes close to every training point, but wiggles between "
                "them: it memorised the noise (overfitting).",
            ),
        )
    if degree < best and test[chosen] > 1.2 * min(test):
        return (
            Notice.info(
                "underfitting",
                f"{errors} Such a simple curve can't follow the sine: the error is high on both "
                "the training and the test data (underfitting).",
            ),
        )
    return (
        Notice.info(
            "good_fit", f"{errors} This degree follows the sine's shape without the noise."
        ),
    )


def _read_learning_rate(chosen: LossCurve, lowest: float) -> tuple[Notice, ...]:
    """A leitura da curva da taxa escolhida, comparada com a loss no fundo da taça."""
    rate = format_value(chosen.rate)
    losses = chosen.losses
    if chosen.diverged:
        return (
            Notice.warning(
                "diverged",
                f"With rate {rate}, each step jumps over the bottom of the bowl and lands higher "
                "on the other side: the loss blows up, and the training stops.",
            ),
        )
    if len(losses) >= 2 and losses[-1] > losses[0]:
        return (
            Notice.warning(
                "growing",
                f"With rate {rate}, the loss is rising: the steps are too big and the path zigzags "
                "wider and wider. With more epochs, it would blow up.",
            ),
        )
    if losses and losses[-1] > 1.05 * lowest:
        return (
            Notice.info(
                "slow",
                f"With rate {rate}, the steps are small: after {len(losses)} epochs the path still "
                "hasn't reached the bottom of the bowl (the cross).",
            ),
        )
    return (
        Notice.info(
            "converging",
            f"With rate {rate}, the path goes down the bowl to the bottom (the cross). Notice that "
            "it doesn't go straight: at each point it follows the steepest direction.",
        ),
    )


def _read_knn(
    k: int, ks: tuple[int, ...], train: list[float], test: list[float]
) -> tuple[Notice, ...]:
    chosen = ks.index(k)
    best = ks[int(np.argmax(test))]
    accuracies = (
        f"With k = {k}, the accuracy is {format_percent(train[chosen])} on the training data and "
        f"{format_percent(test[chosen])} on the test data; the best test is with k = {best}."
    )
    if k < best / 2:
        memorised = " and an almost perfect training score" if train[chosen] >= 0.98 else ""
        return (
            Notice.warning(
                "k_small",
                f"{accuracies} With few neighbours, the boundary follows the noise: islands "
                f"around the isolated points{memorised} (overfitting).",
            ),
        )
    if k > 3 * best:
        return (
            Notice.info(
                "k_large",
                f"{accuracies} With so many neighbours, the boundary gets too smooth and wipes out "
                "the shape of the moons (underfitting).",
            ),
        )
    return (
        Notice.info("k_fine", f"{accuracies} The boundary follows the moons without the noise."),
    )


def _read_regularization(
    kind: str, alpha: float, alphas: NDArray[np.float64], test_r2: list[float], chosen: Any
) -> tuple[Notice, ...]:
    best = float(alphas[int(np.argmax(test_r2))])
    nearest = int(np.argmin(np.abs(np.log10(alphas) - math.log10(alpha))))
    message = (
        f"With alpha = {format_value(alpha)}, the test R² is "
        f"{format_number(test_r2[nearest], 3)}; the best alpha on this data is around "
        f"{format_value(best)}."
    )
    if kind == "lasso":
        zeros = int(np.sum(np.abs(np.asarray(chosen)) < 1e-10))
        message += f" {zeros} of the 8 weights ended up exactly at zero (only 3 features count)."
    return (Notice.info("regularization", message),)


def _read_threshold(
    threshold: float,
    precision: float | None,
    recall: float,
    accuracy: float,
    share: float,
    *,
    best: float,
) -> tuple[Notice, ...]:
    """A leitura do limiar escolhido: o que se troca por o quê, e porque a exatidão engana."""
    limit = format_percent(threshold, 0)
    trap = Notice.info(
        "accuracy_trap",
        f"The accuracy is {format_percent(accuracy)}, but answering “negative” for every row "
        f"would already get {format_percent(1 - share)}: only {format_percent(share)} of the rows "
        "are positive, so accuracy says little here.",
    )
    if precision is None:
        return (
            Notice.warning(
                "flags_nothing",
                f"With a threshold of {limit}, the model flags no row as positive: it never "
                "raises a false alarm, but it misses every positive.",
            ),
            trap,
        )
    reading = (
        f"With a threshold of {limit}, the model finds {format_percent(recall)} of the "
        f"positives (recall), and {format_percent(precision)} of the rows it flags are positive "
        f"(precision); the best F1 on this data is with {format_percent(best, 0)}."
    )
    if recall - precision > 0.15:
        found = Notice.info(
            "threshold_low",
            f"{reading} A low threshold catches almost every case, at the price of many false "
            "alarms.",
        )
    elif precision - recall > 0.15:
        found = Notice.info(
            "threshold_high",
            f"{reading} A high threshold only flags what the model is sure of: few false "
            "alarms, but many missed cases.",
        )
    else:
        found = Notice.info(
            "threshold_balanced", f"{reading} Here, precision and recall are balanced."
        )
    return (found, trap)


def _read_data_size(
    size: int,
    sizes: tuple[int, ...],
    tree_train: NDArray[np.float64],
    tree_test: NDArray[np.float64],
    logistic_test: NDArray[np.float64],
) -> tuple[Notice, ...]:
    chosen = sizes.index(size)
    reading = (
        f"With {size} training rows, the tree gets {format_percent(tree_train[chosen])} of them "
        f"right and {format_percent(tree_test[chosen])} of the test; the logistic regression "
        f"gets {format_percent(logistic_test[chosen])} of the test."
    )
    if tree_test[chosen] < logistic_test[chosen]:
        return (
            Notice.info(
                "simple_wins",
                f"{reading} With so few examples, the tree's questions follow chance: the "
                "simpler model generalises better.",
            ),
        )
    return (
        Notice.info(
            "flexible_wins",
            f"{reading} With enough examples, the tree's boundary follows the moons and beats "
            "the logistic regression, whose straight line can't follow the curves however many "
            "rows it gets (bias).",
        ),
    )
