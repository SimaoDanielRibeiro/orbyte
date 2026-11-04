"""Importância por permutação: quanto o modelo piora quando uma feature deixa de fazer sentido.

Para cada feature, baralham-se os seus valores no teste (a coluna fica com os valores de
outras linhas) e mede-se outra vez a métrica principal. Se o modelo usava essa feature,
piora; se não usava, fica igual. Repete-se algumas vezes, porque o baralhar é ao acaso.

Funciona com qualquer modelo (só precisa de prever) e mede no teste, por isso diz o que
o modelo usa para generalizar, e não o que decorou. Cuidados: com features
correlacionadas, baralhar uma não custa muito, porque a outra a substitui; e cada
coluna depois da preparação conta sozinha (as colunas 0/1 de uma categoria contam
separadamente).
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.evaluation.validation import primary_metric, score
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.models.base import LabModel

REPEATS: Final = 5
"""Quantas vezes se baralha cada feature."""

MAX_ROWS: Final = 2_000
"""Linhas do teste usadas, no máximo (uma amostra, sempre a mesma)."""


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    """O quanto a métrica principal piora quando se baralha uma feature (média e desvio)."""

    feature: str
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class PermutationImportance:
    """A importância de cada feature, da mais para a menos importante.

    Attributes:
        metric_label: A métrica principal (a que se mede).
        as_percent: Se a métrica se mostra como percentagem.
        baseline: A métrica no teste, sem baralhar nada.
        importances: Uma por feature, por ordem decrescente.
    """

    metric_label: str
    as_percent: bool
    baseline: float
    importances: tuple[FeatureImportance, ...]


def permutation_importance(
    model: LabModel[Any], dataset: Dataset, *, seed: int, repeats: int = REPEATS
) -> PermutationImportance:
    """A importância por permutação de cada feature, medida no teste.

    Raises:
        ModelError: Numa tarefa sem alvo.
    """
    if dataset.test.y is None or not dataset.task_type.requires_target:
        raise ModelError("Permutation importance measures predictions against a target.")
    rng = np.random.default_rng(seed)
    X, y = _sample(dataset.test.X, dataset.test.y, rng)
    metric = primary_metric(dataset.task_type)
    baseline = score(model, X, y, dataset)
    sign = 1.0 if metric.higher_is_better is not False else -1.0
    importances = []
    for column, name in enumerate(dataset.feature_names):
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            drops.append(sign * (baseline - score(model, shuffled, y, dataset)))
        importances.append(FeatureImportance(name, float(np.mean(drops)), float(np.std(drops))))
    importances.sort(key=lambda item: item.mean, reverse=True)
    return PermutationImportance(
        metric_label=metric.display_name,
        as_percent=metric.as_percent,
        baseline=baseline,
        importances=tuple(importances),
    )


def _sample(
    X: NDArray[np.float64], y: NDArray[Any], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    if len(X) <= MAX_ROWS:
        return X, y
    chosen = np.sort(rng.choice(len(X), size=MAX_ROWS, replace=False))
    return X[chosen], y[chosen]
