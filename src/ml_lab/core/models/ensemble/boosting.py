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
        title="Number of trees (stages)",
        description=(
            "How many trees are added, one at a time. Unlike the random forest, too many "
            "trees end up memorising the training data."
        ),
        json_schema_extra={"x-search": {"values": [50, 100, 200, 300]}},
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        title="Learning rate",
        description=(
            "How much each new tree counts. Small rates need more trees, but tend to "
            "generalise better."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.01, 0.3]}},
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        title="Depth of each tree",
        description="Small trees (2 to 5 levels) are enough: the power comes from combining them.",
        json_schema_extra={"x-search": {"values": [2, 3, 4, 5]}},
    )
    subsample: float = Field(
        default=1.0,
        ge=0.1,
        le=1.0,
        title="Share of rows per tree",
        description=(
            "Below 1, each tree sees only a sample of the rows (stochastic gradient "
            "boosting): more noise in each tree, less overfitting in the ensemble."
        ),
        json_schema_extra={"x-search": {"range": [0.6, 1.0]}},
    )


class AdaBoostParams(Hyperparameters):
    """Hiperparâmetros do AdaBoost."""

    n_estimators: int = Field(
        default=50,
        ge=10,
        le=500,
        title="Number of models (stages)",
        description="How many weak models (one-level trees) are combined, at most.",
        json_schema_extra={"x-search": {"values": [25, 50, 100, 200]}},
    )
    learning_rate: float = Field(
        default=1.0,
        ge=0.01,
        le=5.0,
        title="Learning rate",
        description="How much each new model counts: smaller needs more models, but is steadier.",
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.05, 2.0]}},
    )


def boosting_state(estimator: Any) -> dict[str, Any]:
    """Quantas etapas o modelo usa e a importância de cada feature."""
    return {
        "n_stages": len(estimator.estimators_),
        "feature_importances": estimator.feature_importances_.tolist(),
    }
