"""Métricas de clustering: quão separados e compactos ficaram os grupos.

Sem respostas certas, mede-se a forma dos grupos no espaço das features (silhueta,
Davies-Bouldin, Calinski-Harabasz). Se houver uma coluna de referência (por exemplo, a
espécie no Iris), o índice de Rand ajustado compara os grupos com ela. Os pontos de
ruído (grupo -1, no DBSCAN) ficam de fora das medidas de forma.
"""

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics as sk

from ml_lab.core.evaluation.metrics.base import MetricInputs, MetricTerm, metric
from ml_lab.core.foundation.types import TaskType

CLUSTERING = (TaskType.CLUSTERING,)

SILHOUETTE_SAMPLE: Final = 2_000
"""A silhueta compara todos os pares de pontos: acima disto, usa uma amostra (sempre a mesma)."""


def _grouped(inputs: MetricInputs) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    """As features e os grupos sem o ruído, se houver pelo menos dois grupos."""
    if inputs.X is None:
        return None
    labels = np.asarray(inputs.predicted, dtype=np.int64)
    kept = labels != -1
    X, labels = inputs.X[kept], labels[kept]
    n_groups = len(np.unique(labels))
    if n_groups < 2 or n_groups >= len(labels):
        return None
    return X, labels


@metric(
    "silhouette",
    display_name="Silhueta",
    task_types=CLUSTERING,
    higher_is_better=True,
    needs=("X", "y_pred"),
)
def silhouette(inputs: MetricInputs) -> float:
    """Para cada ponto, quão mais perto está do seu grupo do que do grupo vizinho (média)."""
    grouped = _grouped(inputs)
    if grouped is None:
        return math.nan
    X, labels = grouped
    sample = SILHOUETTE_SAMPLE if len(X) > SILHOUETTE_SAMPLE else None
    return float(sk.silhouette_score(X, labels, sample_size=sample, random_state=0))


@metric(
    "davies_bouldin",
    display_name="Davies-Bouldin",
    task_types=CLUSTERING,
    higher_is_better=False,
    needs=("X", "y_pred"),
)
def davies_bouldin(inputs: MetricInputs) -> float:
    """A semelhança média de cada grupo com o mais parecido (dispersão sobre distância)."""
    grouped = _grouped(inputs)
    return math.nan if grouped is None else float(sk.davies_bouldin_score(*grouped))


@metric(
    "calinski_harabasz",
    display_name="Calinski-Harabasz",
    task_types=CLUSTERING,
    higher_is_better=True,
    needs=("X", "y_pred"),
)
def calinski_harabasz(inputs: MetricInputs) -> float:
    """A variância entre grupos sobre a variância dentro dos grupos."""
    grouped = _grouped(inputs)
    return math.nan if grouped is None else float(sk.calinski_harabasz_score(*grouped))


@metric(
    "n_clusters",
    display_name="Grupos",
    task_types=CLUSTERING,
    higher_is_better=None,
    needs=("y_pred",),
)
def n_clusters(inputs: MetricInputs) -> float:
    """Quantos grupos têm pontos (sem contar o ruído)."""
    return float(len(set(np.asarray(inputs.predicted).tolist()) - {-1}))


def _noise_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    labels = np.asarray(inputs.predicted)
    return (
        MetricTerm(r"n_{\text{ruído}}", "Pontos sem grupo", float(np.sum(labels == -1))),
        MetricTerm("n", "Pontos avaliados", float(len(labels))),
    )


@metric(
    "noise_ratio",
    display_name="Ruído",
    task_types=CLUSTERING,
    higher_is_better=None,
    needs=("y_pred",),
    as_percent=True,
    breakdown=_noise_terms,
)
def noise_ratio(inputs: MetricInputs) -> float:
    """A proporção de pontos que não ficaram em nenhum grupo."""
    labels = np.asarray(inputs.predicted)
    return float(np.mean(labels == -1)) if len(labels) else math.nan


@metric(
    "adjusted_rand",
    display_name="Rand ajustado",
    task_types=CLUSTERING,
    higher_is_better=True,
)
def adjusted_rand(inputs: MetricInputs) -> float:
    """O acordo entre os grupos e a coluna de referência, descontado o acaso."""
    return float(sk.adjusted_rand_score(inputs.truth, inputs.predicted))
