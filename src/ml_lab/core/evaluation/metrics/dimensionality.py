"""Métricas de redução de dimensionalidade: o que a projeção guardou da estrutura dos dados."""

import math
from typing import Final

import numpy as np
from sklearn.manifold import trustworthiness as sk_trustworthiness

from ml_lab.core.evaluation.metrics.base import MetricInputs, metric
from ml_lab.core.foundation.types import TaskType

NEIGHBORS: Final = 5
"""Quantos vizinhos de cada ponto se comparam antes e depois da projeção."""

SAMPLE: Final = 1_000
"""A confiança compara todos os pares de pontos: acima disto, usa uma amostra (sempre a mesma)."""


@metric(
    "trustworthiness",
    display_name="Confiança dos vizinhos",
    task_types=(TaskType.DIMENSIONALITY_REDUCTION,),
    higher_is_better=True,
    needs=("X", "embedding"),
    as_percent=True,
)
def trustworthiness(inputs: MetricInputs) -> float:
    """Se os vizinhos de cada ponto na projeção eram mesmo vizinhos nos dados originais."""
    if inputs.X is None or inputs.embedding is None or len(inputs.X) <= 2 * NEIGHBORS:
        return math.nan
    X, embedding = inputs.X, inputs.embedding
    if len(X) > SAMPLE:
        chosen = np.random.default_rng(0).choice(len(X), size=SAMPLE, replace=False)
        X, embedding = X[chosen], embedding[chosen]
    return float(sk_trustworthiness(X, embedding, n_neighbors=NEIGHBORS))
