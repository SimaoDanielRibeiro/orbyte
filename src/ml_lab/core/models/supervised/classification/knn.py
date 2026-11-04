"""K vizinhos mais próximos (scikit-learn).

Não aprende pesos: guarda os exemplos de treino e, para cada ponto novo, procura os k
mais próximos e dá a classe mais votada entre eles. A versão didática, com as contas à
vista, está em ``knn_numpy.py``. Ficha didática: ``docs/algorithms/knn.md``.
"""

from typing import Any, Literal, override

from pydantic import Field
from sklearn.neighbors import KNeighborsClassifier

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext


class KnnParams(Hyperparameters):
    """Hiperparâmetros do KNN."""

    n_neighbors: int = Field(
        default=5,
        ge=1,
        le=100,
        title="Number of neighbours (k)",
        description=(
            "How many neighbours vote. Few: the boundary follows every point, noise included "
            "(overfitting); many: it gets smooth and can wipe out details."
        ),
        json_schema_extra={"x-search": {"range": [1, 30]}},
    )
    weights: Literal["uniform", "distance"] = Field(
        default="uniform",
        title="Vote weights",
        description="With “by distance”, the closest neighbours count more.",
        json_schema_extra={
            "x-labels": {"uniform": "Equal", "distance": "By distance"},
            "x-search": {"values": ["uniform", "distance"]},
        },
    )
    metric: Literal["euclidean", "manhattan"] = Field(
        default="euclidean",
        title="Distance",
        description=(
            "Euclidean is the straight line; Manhattan adds up the differences in each "
            "feature, like walking around city blocks."
        ),
        json_schema_extra={
            "x-labels": {"euclidean": "Euclidean", "manhattan": "Manhattan"},
            "x-search": {"values": ["euclidean", "manhattan"]},
        },
    )


def check_neighbors(n_neighbors: int, n_samples: int) -> None:
    """Confirma que há exemplos de treino suficientes para ``n_neighbors`` vizinhos.

    Raises:
        ModelError: Se houver menos exemplos do que vizinhos.
    """
    if n_neighbors > n_samples:
        raise ModelError(
            f"You asked for {n_neighbors} neighbours, but the training data has only "
            f"{n_samples} rows: pick a smaller k."
        )


@MODELS.register("knn")
class Knn(SklearnModel[KnnParams]):
    """K vizinhos mais próximos do scikit-learn."""

    info = ModelInfo(
        name="knn",
        display_name="K-nearest neighbours (KNN)",
        summary=(
            "Classifies each point by the class of the k most similar training examples. It "
            "learns no weights: it keeps the examples and compares."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.INSTANCE_BASED,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=True,
        hyperparameters=KnnParams,
        doc="knn.md",
        keeps_training_rows=True,
    )

    @override
    def build_estimator(self) -> KNeighborsClassifier:
        params = self.hyperparameters
        return KNeighborsClassifier(
            n_neighbors=params.n_neighbors, weights=params.weights, metric=params.metric
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        check_neighbors(self.hyperparameters.n_neighbors, len(X))
        super()._fit(X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return {
            "classes": self.estimator_.classes_.tolist(),
            "n_train": int(self.estimator_.n_samples_fit_),
        }
