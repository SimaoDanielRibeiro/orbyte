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
        title="Proporção de anomalias esperada",
        description=(
            "Que fração do treino se espera que seja anómala: põe o limiar de forma a marcar "
            "essa fração. Não muda a pontuação de cada ponto, só onde se corta."
        ),
    )


class IsolationForestParams(ContaminationParams):
    """Hiperparâmetros do Isolation Forest."""

    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        title="Número de árvores",
        description="Quantas árvores de isolamento fazem a média da pontuação.",
    )


class OneClassSVMParams(Hyperparameters):
    """Hiperparâmetros do One-Class SVM."""

    nu: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        title="Nu (fração fora da fronteira)",
        description=(
            "O máximo de pontos de treino que podem ficar fora da fronteira: na prática, a "
            "proporção de anomalias esperada."
        ),
    )
    kernel: Literal["rbf", "linear", "poly"] = Field(
        default="rbf",
        title="Kernel",
        description="A forma da fronteira à volta dos dados normais.",
        json_schema_extra={
            "x-labels": {"rbf": "RBF (gaussiano)", "linear": "Linear", "poly": "Polinomial"}
        },
    )
    gamma: float = Field(
        default=0.5,
        ge=0.001,
        le=100,
        title="Gamma (alcance de cada ponto)",
        description="No kernel RBF: grande, a fronteira abraça cada ponto; pequeno, é suave.",
        json_schema_extra={"x-scale": "log"},
    )


class LOFParams(ContaminationParams):
    """Hiperparâmetros do LOF."""

    n_neighbors: int = Field(
        default=20,
        ge=2,
        le=100,
        title="Número de vizinhos",
        description=(
            "Quantos vizinhos definem a "
            "densidade local de cada ponto. Poucos: sensível ao ruído; muitos: perde as "
            "anomalias em grupos pequenos."
        ),
    )


@MODELS.register("isolation_forest")
class IsolationForest(SklearnModel[IsolationForestParams]):
    """Isolation Forest do scikit-learn."""

    info = ModelInfo(
        name="isolation_forest",
        display_name="Isolation Forest",
        summary=(
            "Isola cada ponto com cortes ao acaso: as anomalias, por estarem sozinhas, "
            "precisam de poucos cortes. Rápido e bom ponto de partida."
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
            "Desenha a fronteira mais justa à volta da maior parte dos dados; o que fica de "
            "fora é anómalo. Sensível à escala das features."
        ),
        task_types=ANOMALY,
        family=ModelFamily.KERNEL,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=OneClassSVMParams,
        doc="one_class_svm.md",
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
            "Compara a densidade à volta de cada ponto com a dos vizinhos: encontra anomalias "
            "locais, mesmo quando os grupos têm densidades diferentes."
        ),
        task_types=ANOMALY,
        family=ModelFamily.INSTANCE_BASED,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=LOFParams,
        doc="lof.md",
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
