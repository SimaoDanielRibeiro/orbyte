"""O que os modelos de boosting partilham: os hiperparâmetros e o estado.

Boosting é treinar modelos fracos (árvores pequenas) em sequência, cada um a corrigir os
erros dos anteriores. O *gradient boosting* ajusta cada árvore nova aos resíduos (o
gradiente da loss); o AdaBoost dá mais peso aos exemplos que os anteriores falharam.
"""

from typing import Any

from pydantic import Field

from ml_lab.core.schemas.hyperparameters import Hyperparameters


class GradientBoostingParams(Hyperparameters):
    """Hiperparâmetros do gradient boosting."""

    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        title="Número de árvores (etapas)",
        description=(
            "Quantas árvores se juntam, uma de cada vez. Ao contrário da Random Forest, "
            "árvores a mais acabam por decorar o treino."
        ),
        json_schema_extra={"x-search": {"values": [50, 100, 200, 300]}},
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        title="Taxa de aprendizagem",
        description=(
            "Quanto conta cada árvore nova. Taxas pequenas precisam de mais árvores, mas "
            "costumam generalizar melhor."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.01, 0.3]}},
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        title="Profundidade de cada árvore",
        description="Árvores pequenas (de 2 a 5 níveis) bastam: o poder vem de as juntar.",
        json_schema_extra={"x-search": {"values": [2, 3, 4, 5]}},
    )
    subsample: float = Field(
        default=1.0,
        ge=0.1,
        le=1.0,
        title="Fração das linhas por árvore",
        description=(
            "Abaixo de 1, cada árvore vê só uma amostra das linhas (stochastic gradient "
            "boosting): mais ruído em cada árvore, menos overfitting no conjunto."
        ),
        json_schema_extra={"x-search": {"range": [0.6, 1.0]}},
    )


class AdaBoostParams(Hyperparameters):
    """Hiperparâmetros do AdaBoost."""

    n_estimators: int = Field(
        default=50,
        ge=10,
        le=500,
        title="Número de modelos (etapas)",
        description="Quantos modelos fracos (árvores de um só nível) se juntam, no máximo.",
        json_schema_extra={"x-search": {"values": [25, 50, 100, 200]}},
    )
    learning_rate: float = Field(
        default=1.0,
        ge=0.01,
        le=5.0,
        title="Taxa de aprendizagem",
        description="Quanto conta cada modelo novo: menor exige mais modelos, mas é mais estável.",
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.05, 2.0]}},
    )


def boosting_state(estimator: Any) -> dict[str, Any]:
    """Quantas etapas o modelo usa e a importância de cada feature."""
    return {
        "n_stages": len(estimator.estimators_),
        "feature_importances": estimator.feature_importances_.tolist(),
    }
