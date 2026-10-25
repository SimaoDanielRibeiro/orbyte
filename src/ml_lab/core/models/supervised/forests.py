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
        title="Número de árvores",
        description=(
            "Quantas árvores votam. Mais árvores nunca causam overfitting; só tornam o treino "
            "mais lento, e a partir de certo ponto deixam de ajudar."
        ),
        json_schema_extra={"x-search": {"values": [50, 100, 200]}},
    )
    max_depth: int = Field(
        default=0,
        ge=0,
        le=30,
        title="Profundidade máxima",
        description=(
            "Profundidade de cada árvore (0 = sem limite). Numa floresta, árvores fundas "
            "são normais: a média de muitas árvores diferentes corrige os excessos de cada uma."
        ),
        json_schema_extra={"x-search": {"values": [0, 5, 10, 20]}},
    )
    max_features: Literal["sqrt", "log2", "all"] = Field(
        default="sqrt",
        title="Features por divisão",
        description=(
            "Quantas features, escolhidas ao acaso, cada nó pode experimentar. Menos features "
            "tornam as árvores mais diferentes entre si, e a média melhor."
        ),
        json_schema_extra={
            "x-labels": {
                "sqrt": "Raiz quadrada do total",
                "log2": "Logaritmo do total",
                "all": "Todas",
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
