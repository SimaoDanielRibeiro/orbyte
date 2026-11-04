"""Regressão por vetores de suporte (SVR) (scikit-learn).

A ideia do SVM aplicada a números: procura uma função tão plana quanto possível que
passe a menos de ``epsilon`` de cada ponto de treino. Os erros dentro desse "tubo" não
contam; os pontos fora dele (ou na sua borda) são os vetores de suporte. Ficha
didática: ``docs/algorithms/svm.md``.
"""

from typing import Any, override

from pydantic import Field
from sklearn.svm import SVR

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import MODELS, Implementation, ModelFamily, ModelInfo
from ml_lab.core.models.supervised.kernels import KernelParams


class SvrParams(KernelParams):
    """Hiperparâmetros do SVR."""

    epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=10,
        title="Epsilon (tube width)",
        description=(
            "Errors smaller than this don't count. On the target's scale: with a target in the "
            "thousands, an epsilon of 0.1 is practically zero."
        ),
    )


@MODELS.register("svr")
class Svr(SklearnModel[SvrParams]):
    """SVR do scikit-learn."""

    info = ModelInfo(
        name="svr",
        display_name="Support vector regression (SVR)",
        summary=(
            "A function as flat as possible, with a margin of tolerance for errors; with the RBF "
            "kernel, it follows curves."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.KERNEL,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=SvrParams,
        doc="svm.md",
        keeps_training_rows=True,
    )

    @override
    def build_estimator(self) -> SVR:
        params = self.hyperparameters
        return SVR(C=params.C, kernel=params.kernel, gamma=params.gamma, epsilon=params.epsilon)

    @override
    def state(self) -> dict[str, Any]:
        return {
            "n_support": len(self.estimator_.support_),
            "support_indices": self.estimator_.support_.tolist(),
        }
