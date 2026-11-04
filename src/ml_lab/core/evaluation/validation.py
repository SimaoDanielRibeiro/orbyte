"""Validação cruzada e curvas de aprendizagem: o resultado do teste não foi sorte?

Um único teste pode sair melhor ou pior por acaso (as linhas que calharam no teste). A
**validação cruzada** parte o treino em k folds e treina k vezes, cada uma a avaliar
num fold diferente: a média diz o desempenho esperado e o desvio padrão diz quanto ele
varia. As transformações aprendem de novo em cada fold (``prepare_folds``), por isso o
fold de avaliação nunca influencia o treino.

A **curva de aprendizagem** treina com frações crescentes do treino de cada fold e
compara o desempenho no treino e na avaliação: se as curvas ainda se aproximam no fim,
mais dados ajudariam; se ficam afastadas, o modelo decora (variância); se ficam juntas
mas baixas, o modelo é simples demais (viés).

As duas medem a métrica principal da tarefa. A validação cruzada pode medir outra (os
agentes da fase 9 otimizam a que se lhes pede, como a F1 ou a ROC AUC).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.metrics.base import METRICS, Metric, MetricInputs
from ml_lab.core.foundation.errors import CapabilityError, MLLabError, ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import LabModel, class_probabilities
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.formatting import format_count, format_number, format_percent

FRACTIONS: Final = (0.1, 0.25, 0.5, 0.75, 1.0)
"""As frações do treino de cada fold usadas na curva de aprendizagem."""

GAP: Final = 0.1
"""Diferença entre treino e avaliação, na métrica principal, que já indica variância a mais."""


@dataclass(frozen=True, slots=True)
class FoldScore:
    """A métrica principal num fold: no treino desse fold e no fold deixado de fora."""

    fold: int
    train: float | None
    test: float | None


@dataclass(frozen=True, slots=True)
class CrossValidation:
    """O resultado da validação cruzada.

    Attributes:
        metric: O nome da métrica principal.
        metric_label: O nome a mostrar.
        as_percent: Se se mostra como percentagem.
        folds: O resultado de cada fold.
        mean: A média na avaliação.
        std: O desvio padrão na avaliação.
        notices: A leitura didática do resultado.
    """

    metric: str
    metric_label: str
    as_percent: bool
    folds: tuple[FoldScore, ...]
    mean: float | None
    std: float | None
    notices: tuple[Notice, ...] = ()


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """Um ponto da curva de aprendizagem: o tamanho do treino e as médias (e desvios)."""

    size: int
    train_mean: float
    train_std: float
    test_mean: float
    test_std: float


@dataclass(frozen=True, slots=True)
class LearningCurve:
    """A curva de aprendizagem."""

    metric: str
    metric_label: str
    as_percent: bool
    points: tuple[CurvePoint, ...]
    notices: tuple[Notice, ...] = ()


def primary_metric(task_type: TaskType) -> Metric:
    """A métrica principal de ``task_type``.

    Raises:
        ModelError: Se o tipo de tarefa não tiver métrica principal.
    """
    name = PRIMARY_METRIC.get(task_type)
    if name is None:
        raise ModelError(f"{task_type.label} has no main metric.")
    return METRICS.get(name)


def score(
    model: LabModel[Any],
    X: NDArray[np.float64],
    y: NDArray[Any],
    dataset: Dataset,
    metric: Metric | None = None,
) -> float:
    """Uma métrica do modelo em ``X`` (só para tarefas com alvo); por omissão, a principal.

    As métricas que precisam das probabilidades (a ROC AUC, a log-loss) recebem uma coluna
    por classe do dataset, mesmo que alguma não tenha aparecido no treino deste fold.
    """
    chosen = metric or primary_metric(dataset.task_type)
    proba = class_probabilities(model, X, dataset.n_classes) if chosen.needs_proba else None
    inputs = MetricInputs(
        y_true=y, y_pred=model.predict(X), y_proba=proba, n_classes=dataset.n_classes
    )
    return chosen(inputs)


def cross_validate(
    model_class: type[LabModel[Any]],
    hyperparameters: Hyperparameters,
    seed: int,
    folds: Sequence[Dataset],
    metric: Metric | None = None,
    on_fold: Callable[[int], None] | None = None,
) -> CrossValidation:
    """Treina e avalia em cada fold.

    Args:
        model_class: O modelo.
        hyperparameters: Os hiperparâmetros, os mesmos em todos os folds.
        seed: A seed do modelo.
        folds: Um ``Dataset`` por fold (de ``prepare_folds``): o "teste" é o fold de fora.
        metric: A métrica a medir; por omissão, a principal da tarefa.
        on_fold: Chamado com o número de cada fold (1, 2…) antes de o treinar, para quem
            acompanha saber em que fold vai (os agentes da fase 9).

    Raises:
        ModelError: Numa tarefa sem alvo (a validação cruzada mede previsões), ou com uma
            métrica que não é da tarefa.
        CapabilityError: Se a métrica precisar de probabilidades e o modelo não as der.
    """
    task = _supervised(folds)
    metric = metric or primary_metric(task)
    if task not in metric.task_types:
        raise ModelError(f"{metric.display_name} isn't a {task.label.lower()} metric.")
    if metric.needs_proba and not model_class.info.supports_proba:
        raise CapabilityError(
            f"{metric.display_name} needs probabilities, and "
            f"'{model_class.info.display_name}' doesn't give them."
        )
    results = []
    for number, dataset in enumerate(folds, start=1):
        if on_fold is not None:
            on_fold(number)
        model = model_class(hyperparameters, seed=seed).fit(dataset.train.X, dataset.train.y)
        train = _finite(score(model, dataset.train.X, _target(dataset.train.y), dataset, metric))
        test = _finite(score(model, dataset.test.X, _target(dataset.test.y), dataset, metric))
        results.append(FoldScore(number, train, test))
    tests = [fold.test for fold in results if fold.test is not None]
    mean = float(np.mean(tests)) if tests else None
    std = float(np.std(tests)) if tests else None
    return CrossValidation(
        metric=metric.name,
        metric_label=metric.display_name,
        as_percent=metric.as_percent,
        folds=tuple(results),
        mean=mean,
        std=std,
        notices=_read_cross_validation(metric, results, mean, std),
    )


def learning_curve(
    model_class: type[LabModel[Any]],
    hyperparameters: Hyperparameters,
    seed: int,
    folds: Sequence[Dataset],
    fractions: Sequence[float] = FRACTIONS,
) -> LearningCurve:
    """A métrica no treino e na avaliação, com frações crescentes do treino de cada fold.

    Raises:
        ModelError: Numa tarefa sem alvo.
    """
    task = _supervised(folds)
    metric = primary_metric(task)
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(dataset.train)) for dataset in folds]
    points = []
    for fraction in fractions:
        trains, tests, sizes = [], [], []
        for dataset, order in zip(folds, orders, strict=True):
            target = _target(dataset.train.y)
            chosen = _subset(order, target, fraction, task)
            try:
                model = model_class(hyperparameters, seed=seed).fit(
                    dataset.train.X[chosen], target[chosen]
                )
            except MLLabError:  # poucas linhas para este modelo (menos vizinhos do que k...)
                continue
            trains.append(score(model, dataset.train.X[chosen], target[chosen], dataset))
            tests.append(score(model, dataset.test.X, _target(dataset.test.y), dataset))
            sizes.append(len(chosen))
        trains, tests = _finite_values(trains), _finite_values(tests)
        if trains and tests:
            points.append(
                CurvePoint(
                    size=int(np.mean(sizes)),
                    train_mean=float(np.mean(trains)),
                    train_std=float(np.std(trains)),
                    test_mean=float(np.mean(tests)),
                    test_std=float(np.std(tests)),
                )
            )
    return LearningCurve(
        metric=metric.name,
        metric_label=metric.display_name,
        as_percent=metric.as_percent,
        points=tuple(points),
        notices=_read_learning_curve(metric, points),
    )


def _supervised(folds: Sequence[Dataset]) -> TaskType:
    if not folds:
        raise ModelError("There are no folds to evaluate.")
    task = folds[0].task_type
    if not task.requires_target:
        raise ModelError(
            f"Cross-validation measures predictions against a target: {task.label.lower()} "
            "has none."
        )
    return task


def _target(y: NDArray[Any] | None) -> NDArray[Any]:
    if y is None:
        raise ModelError("Cross-validation needs the target.")
    return y


def _subset(order: NDArray[np.intp], y: NDArray[Any], fraction: float, task: TaskType) -> Any:
    """As primeiras linhas (já baralhadas); na classificação, uma fração de cada classe."""
    if not task.is_classification:
        return np.sort(order[: max(2, math.ceil(fraction * len(order)))])
    chosen = []
    for value in np.unique(y):
        members = order[y[order] == value]
        chosen.append(members[: max(1, math.ceil(fraction * len(members)))])
    return np.sort(np.concatenate(chosen))


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _finite_values(values: Sequence[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _written(metric: Metric, value: float) -> str:
    return format_percent(value) if metric.as_percent else format_number(value, 3)


def _read_cross_validation(
    metric: Metric, folds: Sequence[FoldScore], mean: float | None, std: float | None
) -> tuple[Notice, ...]:
    if mean is None or std is None:
        return ()
    label = metric.display_name
    notices = [
        Notice.info(
            "cross_validation",
            f"{label} over {format_count(len(folds))} folds: {_written(metric, mean)} on "
            f"average, varying by ± {_written(metric, std)} from fold to fold. It's a safer "
            "estimate than a single test.",
        )
    ]
    if std > GAP / 2:
        notices.append(
            Notice.warning(
                "unstable",
                f"The result changes a lot from fold to fold (± {_written(metric, std)}): with "
                "so little data, a single test can be quite misleading.",
            )
        )
    return tuple(notices)


def _read_learning_curve(metric: Metric, points: Sequence[CurvePoint]) -> tuple[Notice, ...]:
    if len(points) < 2:
        return ()
    last, before = points[-1], points[-2]
    sign = 1.0 if metric.higher_is_better is not False else -1.0
    gap = sign * (last.train_mean - last.test_mean)
    rising = sign * (last.test_mean - before.test_mean) > 0.01
    if gap > GAP:
        message = (
            f"With all the training data, the training score is {format_number(gap * 100, 1)} "
            f"points above the evaluation ({metric.display_name}): the model memorises part of "
            "the training data (variance). More data, more regularisation or a simpler model "
            "help."
        )
        return (Notice.warning("high_variance", message),)
    if rising:
        message = (
            "The evaluation still improves as the training data grows: with more data, the "
            "model would probably get better."
        )
        return (Notice.info("more_data", message),)
    message = (
        "The training and the evaluation are close and have levelled off: more data won't help "
        "much any more. If the result is weak, the model is too simple for the data (bias): "
        "try a more flexible one or other features."
    )
    return (Notice.info("plateau", message),)
