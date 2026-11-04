"""O que as Random Forests de classificação e de regressão partilham."""

from typing import Any, Literal

from pydantic import Field

from ml_lab.core.schemas.hyperparameters import Hyperparameters


class ForestParams(Hyperparameters):
    """Hiperparâmetros comuns às Random Forests."""

    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        title="Number of trees",
        description=(
            "How many trees vote. More trees never cause overfitting; they only make training "
            "slower, and past a certain point they stop helping."
        ),
        json_schema_extra={"x-search": {"values": [50, 100, 200]}},
    )
    max_depth: int = Field(
        default=0,
        ge=0,
        le=30,
        title="Maximum depth",
        description=(
            "The depth of each tree (0 = no limit). In a forest, deep trees are normal: the "
            "average of many different trees corrects the excesses of each one."
        ),
        # O 0 lê-se pelo nome: "no limit" diz mais do que "0", no formulário e na conversa.
        json_schema_extra={"x-search": {"values": [0, 5, 10, 20]}, "x-labels": {"0": "no limit"}},
    )
    max_features: Literal["sqrt", "log2", "all"] = Field(
        default="sqrt",
        title="Features per split",
        description=(
            "How many features, picked at random, each node can try. Fewer features make the "
            "trees more different from each other, and the average better."
        ),
        json_schema_extra={
            "x-labels": {
                "sqrt": "Square root of the total",
                "log2": "Logarithm of the total",
                "all": "All",
            },
            "x-search": {"values": ["sqrt", "log2", "all"]},
        },
    )


def forest_options(params: ForestParams) -> dict[str, Any]:
    """Os argumentos comuns do estimador do scikit-learn."""
    return {
        "n_estimators": params.n_estimators,
        "max_depth": params.max_depth or None,
        "max_features": None if params.max_features == "all" else params.max_features,
    }


def forest_state(estimator: Any) -> dict[str, Any]:
    """O que a floresta aprendeu: quantas árvores e a importância de cada feature."""
    return {
        "n_trees": len(estimator.estimators_),
        "feature_importances": estimator.feature_importances_.tolist(),
        "oob_score": float(estimator.oob_score_) if hasattr(estimator, "oob_score_") else None,
    }
