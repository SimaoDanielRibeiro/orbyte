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
        title="Número de vizinhos (k)",
        description=(
            "Quantos vizinhos votam. Poucos: a fronteira segue cada ponto, incluindo o ruído "
            "(overfitting); muitos: fica suave e pode apagar detalhes."
        ),
        json_schema_extra={"x-search": {"range": [1, 30]}},
    )
    weights: Literal["uniform", "distance"] = Field(
        default="uniform",
        title="Peso dos votos",
        description="Com «por distância», os vizinhos mais próximos contam mais.",
        json_schema_extra={
            "x-labels": {"uniform": "Iguais", "distance": "Por distância"},
            "x-search": {"values": ["uniform", "distance"]},
        },
    )
    metric: Literal["euclidean", "manhattan"] = Field(
        default="euclidean",
        title="Distância",
        description=(
            "Euclidiana é a linha reta; Manhattan soma as diferenças de cada feature, como "
            "andar por quarteirões."
        ),
        json_schema_extra={
            "x-labels": {"euclidean": "Euclidiana", "manhattan": "Manhattan"},
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
            f"Pediste {n_neighbors} vizinhos, mas o treino só tem {n_samples} linhas: escolhe um "
            "k menor."
        )


@MODELS.register("knn")
class Knn(SklearnModel[KnnParams]):
    """K vizinhos mais próximos do scikit-learn."""

    info = ModelInfo(
        name="knn",
        display_name="K vizinhos mais próximos (KNN)",
        summary=(
            "Classifica cada ponto pela classe dos k exemplos de treino mais parecidos. Não "
            "aprende pesos: guarda os exemplos e compara."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.INSTANCE_BASED,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=True,
        hyperparameters=KnnParams,
        doc="knn.md",
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
