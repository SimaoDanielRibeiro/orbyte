"""Serviço de análise: o que se pergunta a um modelo depois do treino (avaliação e Explicar).

Usa o modelo e os dados que o serviço de experiências guarda dos últimos runs:

- os dados dos gráficos da avaliação (ROC, resíduos, silhueta, dendrograma…);
- a validação cruzada e a curva de aprendizagem (treinam outra vez, nos folds);
- a importância por permutação e a dependência parcial (ecrã Explicar);
- a explicação de uma previsão: o contributo de cada feature para ela.

Os cálculos mais pesados (validação cruzada, curva de aprendizagem, permutação) ficam
guardados por run, para não se repetirem ao voltar ao ecrã.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.evaluation.validation import (
    CrossValidation,
    LearningCurve,
    cross_validate,
    learning_curve,
)
from ml_lab.core.explainability.partial_dependence import PartialDependence, partial_dependence
from ml_lab.core.explainability.permutation import PermutationImportance, permutation_importance
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import MODELS, LabModel, class_probabilities
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import MODEL_SEED, ExperimentService
from ml_lab.core.visualization_data.evaluation import EvaluationCharts, evaluation_charts
from ml_lab.core.visualization_data.learned import linear_weights
from ml_lab.formatting import format_number

CACHE_SIZE: Final = 50
"""Resultados guardados, no máximo (os mais antigos saem primeiro)."""

FOLDS: Final = 5

_HOW_TO_READ: Final[Mapping[Literal["weights", "replace"], str]] = MappingProxyType(
    {
        "weights": (
            "It's a linear model: each bar is the feature's weight times its value in this row, "
            "and the prediction (or the class score) is the intercept plus the sum of the bars. "
            "Positive bars push the prediction up; negative ones, down."
        ),
        "replace": (
            "Each bar is how much the prediction changes because the feature has this value, "
            "instead of the training mean: positive ones push the prediction up, negative ones "
            "down. It's an approximation, because it ignores the interactions between features."
        ),
    }
)
"""Como se leem os contributos, conforme o método que os calculou."""


@dataclass(frozen=True, slots=True)
class Contribution:
    """O contributo de uma feature para uma previsão."""

    feature: str
    value: float
    effect: float


@dataclass(frozen=True, slots=True)
class RowExplanation:
    """Uma previsão explicada.

    Attributes:
        row: A linha do teste (0 é a primeira).
        prediction: O que o modelo previu, escrito.
        actual: O valor real, escrito (se houver).
        probabilities: Na classificação com probabilidades, a de cada classe.
        method: Como se calcularam os contributos: ``"weights"`` (pesos × valores, nos
            modelos lineares, exato) ou ``"replace"`` (trocar a feature pela média do treino
            e ver quanto a previsão muda, em qualquer modelo).
        effect_label: Em que unidades estão os contributos.
        contributions: O contributo de cada feature, do maior para o menor (em valor absoluto).
        base: O ponto de partida (a ordenada, nos modelos lineares).
        how_to_read: Como se leem os contributos (depende do método).
    """

    row: int
    prediction: str
    actual: str | None
    probabilities: tuple[tuple[str, float], ...]
    method: Literal["weights", "replace"]
    effect_label: str
    contributions: tuple[Contribution, ...]
    base: float | None
    how_to_read: str


class AnalysisService:
    """Análises de runs recentes.

    Args:
        experiments: Onde estão o modelo e os dados de cada run.
        datasets: Para preparar os folds da validação cruzada.
    """

    def __init__(self, experiments: ExperimentService, datasets: DatasetService) -> None:
        self._experiments = experiments
        self._datasets = datasets
        self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()

    def charts(self, run_id: str) -> EvaluationCharts:
        """Os dados dos gráficos da avaliação.

        Raises:
            ModelError: Se o run já não estiver em memória.
        """
        artifacts = self._experiments.artifacts(run_id)
        return self._remember(
            ("charts", run_id), lambda: evaluation_charts(artifacts.model, artifacts.dataset)
        )

    def cross_validation(self, run_id: str, k: int = FOLDS) -> CrossValidation:
        """A validação cruzada do modelo do run, com os mesmos hiperparâmetros e a mesma seed.

        Raises:
            ModelError: Numa tarefa sem alvo, ou se o run já não estiver em memória.
        """

        def compute() -> CrossValidation:
            return cross_validate(*self._folds(run_id, k))

        return self._remember(("cv", run_id, k), compute)

    def learning_curve(self, run_id: str, k: int = FOLDS) -> LearningCurve:
        """A curva de aprendizagem do modelo do run, nos folds da validação cruzada.

        Raises:
            ModelError: Numa tarefa sem alvo, ou se o run já não estiver em memória.
        """

        def compute() -> LearningCurve:
            return learning_curve(*self._folds(run_id, k))

        return self._remember(("curve", run_id, k), compute)

    def permutation_importance(self, run_id: str) -> PermutationImportance:
        """A importância por permutação de cada feature, no teste.

        Raises:
            ModelError: Numa tarefa sem alvo, ou se o run já não estiver em memória.
        """
        artifacts = self._experiments.artifacts(run_id)
        seed = _seed(artifacts.config.seed)
        return self._remember(
            ("permutation", run_id),
            lambda: permutation_importance(artifacts.model, artifacts.dataset, seed=seed),
        )

    def partial_dependence(self, run_id: str, feature: str) -> PartialDependence:
        """A dependência parcial de ``feature``, no teste.

        Raises:
            ModelError: Se a feature não existir, numa tarefa sem alvo, ou se o run já não
                estiver em memória.
        """
        artifacts = self._experiments.artifacts(run_id)
        seed = _seed(artifacts.config.seed)
        return self._remember(
            ("dependence", run_id, feature),
            lambda: partial_dependence(artifacts.model, artifacts.dataset, feature, seed=seed),
        )

    def test_rows(self, run_id: str) -> int:
        """Quantas linhas de teste tem o run (para escolher uma previsão a explicar)."""
        return len(self._experiments.artifacts(run_id).dataset.test)

    def explain_row(self, run_id: str, row: int) -> RowExplanation:
        """O contributo de cada feature para a previsão de uma linha do teste.

        Raises:
            ModelError: Se a linha não existir, numa tarefa sem alvo, ou se o run já não
                estiver em memória.
        """
        artifacts = self._experiments.artifacts(run_id)
        model, dataset = artifacts.model, artifacts.dataset
        if not dataset.task_type.requires_target:
            raise ModelError("Predictions can only be explained in tasks with a target.")
        test = dataset.test
        if not 0 <= row < len(test):
            raise ModelError(
                f"The test data has {len(test)} rows: choose one from 0 to {len(test) - 1}."
            )
        x = test.X[row : row + 1]
        predicted = model.predict(x)[0]
        probabilities: tuple[tuple[str, float], ...] = ()
        if dataset.task_type.is_classification and model.info.supports_proba:
            proba = class_probabilities(model, x, dataset.n_classes)[0]
            probabilities = tuple(
                (label, float(p)) for label, p in zip(dataset.class_labels, proba, strict=True)
            )
        actual = None if test.y is None else _written(dataset, test.y[row])
        classification = dataset.task_type.is_classification
        weights = linear_weights(
            model.state(),
            task_type=dataset.task_type,
            feature_names=dataset.feature_names,
            class_labels=dataset.class_labels,
            target_name=dataset.target_name,
        )
        output = None  # a linha de pesos que explica esta previsão (a classe prevista)
        if weights is not None:
            output = weights.row_for(int(predicted)) if classification else 0
        method: Literal["weights", "replace"]
        base: float | None = None
        if weights is not None and output is not None:  # pesos × valores: a decomposição exata
            effects = np.asarray(weights.weights[output]) * x[0]
            base = weights.intercept_for(output)
            if classification:
                label = f"Contribution to the score of “{weights.outputs[output]}”"
            else:
                label = f"Contribution to the prediction of {dataset.target_name or 'the target'}"
            method = "weights"
        else:
            chosen = int(predicted) if classification else None
            effects, label = _replacement_effects(model, dataset, x, chosen)
            method = "replace"
        contributions = sorted(
            (
                Contribution(name, float(value), float(effect))
                for name, value, effect in zip(dataset.feature_names, x[0], effects, strict=True)
            ),
            key=lambda item: abs(item.effect),
            reverse=True,
        )
        return RowExplanation(
            row=row,
            prediction=_written(dataset, predicted),
            actual=actual,
            probabilities=probabilities,
            method=method,
            effect_label=label,
            contributions=tuple(contributions),
            base=base,
            how_to_read=_HOW_TO_READ[method],
        )

    def _folds(
        self, run_id: str, k: int
    ) -> tuple[type[LabModel[Any]], Hyperparameters, int, tuple[Dataset, ...]]:
        """O modelo, os hiperparâmetros, a seed do modelo e os folds de um run."""
        artifacts = self._experiments.artifacts(run_id)
        config = artifacts.config
        if not artifacts.dataset.task_type.requires_target:
            raise ModelError(
                "Cross-validation measures predictions against a target: "
                f"{artifacts.dataset.task_type.label.lower()} has none."
            )
        seed = _seed(config.seed)
        folds = self._datasets.folds(config.dataset, config.preparation, seed, k)
        model_seed = SeedManager(seed).seed_for(MODEL_SEED)
        return MODELS.get(config.model.name), artifacts.model.hyperparameters, model_seed, folds

    def _remember[T](self, key: tuple[Any, ...], compute: Callable[[], T]) -> T:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                cached: T = self._cache[key]
                return cached
        value = compute()
        with self._lock:
            self._cache[key] = value
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return value


def _seed(seed: int | None) -> int:
    assert seed is not None  # o serviço de experiências resolve sempre a seed
    return seed


def _written(dataset: Dataset, value: Any) -> str:
    if dataset.task_type.is_classification:
        return dataset.class_labels[int(value)]
    return format_number(float(value), 3)


def _replacement_effects(
    model: LabModel[Any], dataset: Dataset, x: NDArray[np.float64], predicted: int | None
) -> tuple[NDArray[np.float64], str]:
    """Quanto a previsão muda quando cada feature passa a valer a média do treino."""
    means = dataset.splits[Partition.TRAIN].X.mean(axis=0)
    changed = np.repeat(x, dataset.n_features, axis=0)
    changed[np.arange(dataset.n_features), np.arange(dataset.n_features)] = means
    if dataset.task_type is TaskType.REGRESSION:
        target = dataset.target_name or "the target"
        original = float(model.predict(x)[0])
        return original - np.asarray(model.predict(changed), dtype=np.float64), (
            f"How much the prediction of {target} changes because of this feature's value"
        )
    if model.info.supports_proba and predicted is not None:
        original = float(class_probabilities(model, x, dataset.n_classes)[0, predicted])
        label = dataset.class_labels[predicted]
        return original - class_probabilities(model, changed, dataset.n_classes)[:, predicted], (
            f"How much the probability of “{label}” changes because of this feature's value"
        )
    original_class = model.predict(x)[0]
    changes = (np.asarray(model.predict(changed)) != original_class).astype(np.float64)
    return changes, "1 if swapping this feature for the mean changes the predicted class"
