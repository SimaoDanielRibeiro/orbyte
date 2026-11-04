"""Deteção de anomalias (scikit-learn): Isolation Forest, One-Class SVM e LOF.

Os três dão a cada ponto uma pontuação de anomalia (maior = mais estranho) e marcam como
anomalias os pontos acima de um limiar:

- **Isolation Forest**: árvores que separam pontos com cortes ao acaso; as anomalias
  isolam-se com poucos cortes, porque estão sozinhas;
- **One-Class SVM**: a fronteira mais justa à volta da maior parte dos dados; o que fica
  de fora é anómalo;
- **LOF** (*Local Outlier Factor*): compara a densidade à volta de cada ponto com a dos
  seus vizinhos; um ponto numa zona mais vazia do que os vizinhos é anómalo.

Fichas didáticas: ``docs/algorithms/isolation_forest.md``, ``one_class_svm.md`` e ``lof.md``.
"""

from typing import Any, Literal, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM as SklearnOneClassSVM

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
from ml_lab.core.models.distances import same_rows
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext

ANOMALY = frozenset({TaskType.ANOMALY_DETECTION})


class ContaminationParams(Hyperparameters):
    """A proporção de anomalias esperada, comum ao Isolation Forest e ao LOF."""

    contamination: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        title="Expected share of anomalies",
        description=(
            "What fraction of the training data is expected to be anomalous: it sets the "
            "threshold so it flags that fraction. It doesn't change each point's score, only "
            "where the cut goes."
        ),
    )


class IsolationForestParams(ContaminationParams):
    """Hiperparâmetros do Isolation Forest."""

    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        title="Number of trees",
        description="How many isolation trees average the score.",
    )


class OneClassSVMParams(Hyperparameters):
    """Hiperparâmetros do One-Class SVM."""

    nu: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        title="Nu (share outside the boundary)",
        description=(
            "The most training points that can fall outside the boundary: in practice, the "
            "expected share of anomalies."
        ),
    )
    kernel: Literal["rbf", "linear", "poly"] = Field(
        default="rbf",
        title="Kernel",
        description="The shape of the boundary around the normal data.",
        json_schema_extra={
            "x-labels": {"rbf": "RBF (Gaussian)", "linear": "Linear", "poly": "Polynomial"}
        },
    )
    gamma: float = Field(
        default=0.5,
        ge=0.001,
        le=100,
        title="Gamma (reach of each point)",
        description="In the RBF kernel: large, the boundary hugs each point; small, it's smooth.",
        json_schema_extra={"x-scale": "log"},
    )


class LOFParams(ContaminationParams):
    """Hiperparâmetros do LOF."""

    n_neighbors: int = Field(
        default=20,
        ge=2,
        le=100,
        title="Number of neighbours",
        description=(
            "How many neighbours define each point's local density. Few: sensitive to noise; "
            "many: it misses anomalies in small groups."
        ),
    )


@MODELS.register("isolation_forest")
class IsolationForest(SklearnModel[IsolationForestParams]):
    """Isolation Forest do scikit-learn."""

    info = ModelInfo(
        name="isolation_forest",
        display_name="Isolation Forest",
        summary=(
            "Isolates each point with random cuts: anomalies, being on their own, need few "
            "cuts. Fast, and a good starting point."
        ),
        task_types=ANOMALY,
        family=ModelFamily.ENSEMBLE,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=IsolationForestParams,
        doc="isolation_forest.md",
    )

    @override
    def build_estimator(self) -> SklearnIsolationForest:
        params = self.hyperparameters
        return SklearnIsolationForest(
            n_estimators=params.n_estimators,
            contamination=params.contamination,
            random_state=self.seed,
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        super()._fit(X, None, context)

    @override
    def state(self) -> dict[str, Any]:
        return {"threshold": -float(self.estimator_.offset_)}


@MODELS.register("one_class_svm")
class OneClassSVM(SklearnModel[OneClassSVMParams]):
    """One-Class SVM do scikit-learn."""

    info = ModelInfo(
        name="one_class_svm",
        display_name="One-Class SVM",
        summary=(
            "Draws the tightest boundary around most of the data; whatever falls outside is "
            "anomalous. Sensitive to the scale of the features."
        ),
        task_types=ANOMALY,
        family=ModelFamily.KERNEL,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=OneClassSVMParams,
        doc="one_class_svm.md",
        keeps_training_rows=True,
    )

    @override
    def build_estimator(self) -> SklearnOneClassSVM:
        params = self.hyperparameters
        return SklearnOneClassSVM(nu=params.nu, kernel=params.kernel, gamma=params.gamma)

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        super()._fit(X, None, context)

    @override
    def state(self) -> dict[str, Any]:
        return {
            "threshold": -float(self.estimator_.offset_[0]),
            "n_support": len(self.estimator_.support_),
        }


@MODELS.register("lof")
class LOF(SklearnModel[LOFParams]):
    """Local Outlier Factor do scikit-learn, em modo de deteção de novidades.

    Com ``novelty=True`` sabe pontuar pontos novos. Os pontos de treino ficam com a
    pontuação calculada no treino (``negative_outlier_factor_``): pontuá-los de novo
    contaria cada ponto como vizinho de si próprio.
    """

    info = ModelInfo(
        name="lof",
        display_name="Local Outlier Factor (LOF)",
        summary=(
            "Compares the density around each point with its neighbours': it finds local "
            "anomalies, even when the groups have different densities."
        ),
        task_types=ANOMALY,
        family=ModelFamily.INSTANCE_BASED,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=LOFParams,
        doc="lof.md",
        keeps_training_rows=True,
    )

    def __init__(self, hyperparameters: LOFParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._train = np.zeros((0, 0))

    @override
    def build_estimator(self) -> LocalOutlierFactor:
        params = self.hyperparameters
        return LocalOutlierFactor(
            n_neighbors=params.n_neighbors, contamination=params.contamination, novelty=True
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        super()._fit(X, None, context)
        self._train = X.copy()

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        scores = self._anomaly_scores(X)
        flags: NDArray[np.int64] = (scores > -float(self.estimator_.offset_)).astype(np.int64)
        return flags

    @override
    def _anomaly_scores(self, X: Features) -> NDArray[np.float64]:
        if same_rows(X, self._train):
            return -np.asarray(self.estimator_.negative_outlier_factor_, dtype=np.float64)
        return super()._anomaly_scores(X)

    @override
    def state(self) -> dict[str, Any]:
        return {"threshold": -float(self.estimator_.offset_)}
