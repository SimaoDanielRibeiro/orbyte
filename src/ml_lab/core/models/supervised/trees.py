"""O que as árvores de decisão de classificação e de regressão partilham.

Os hiperparâmetros são os mesmos, e o estado inclui a própria árvore (feature e limiar
de cada nó, e o que cada folha prevê), em valores simples, para a interface a escrever
como regras "se … então …" com os nomes das features.
"""

from typing import Any

import numpy as np
from pydantic import Field

from ml_lab.core.schemas.hyperparameters import Hyperparameters


class TreeParams(Hyperparameters):
    """Hiperparâmetros comuns às árvores de decisão."""

    max_depth: int = Field(
        default=5,
        ge=0,
        le=30,
        title="Profundidade máxima",
        description=(
            "Quantas perguntas seguidas a árvore pode fazer (0 = sem limite). Árvores fundas "
            "decoram o treino (overfitting); árvores rasas podem ficar simples demais."
        ),
        json_schema_extra={"x-search": {"values": [2, 3, 4, 5, 6, 8, 10, 0]}},
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=100,
        title="Mínimo de exemplos por folha",
        description=(
            "Cada folha tem de ter pelo menos este número de exemplos: valores maiores evitam "
            "folhas que existem só para acertar num ponto."
        ),
        json_schema_extra={"x-search": {"values": [1, 2, 5, 10, 20]}},
    )


def tree_state(estimator: Any) -> dict[str, Any]:
    """A árvore em valores simples: a estrutura e a importância de cada feature."""
    tree = estimator.tree_
    # Na regressão, uma coluna (a média); na classificação, uma por classe (as proporções).
    values = np.asarray(tree.value)[:, 0, :]
    return {
        "depth": int(estimator.get_depth()),
        "n_leaves": int(estimator.get_n_leaves()),
        "feature_importances": estimator.feature_importances_.tolist(),
        "tree": {
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "left": tree.children_left.tolist(),
            "right": tree.children_right.tolist(),
            "samples": tree.n_node_samples.tolist(),
            "value": values.tolist(),
        },
    }
