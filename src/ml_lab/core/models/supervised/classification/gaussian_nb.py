"""Naive Bayes gaussiano (scikit-learn).

Para cada classe, aprende a média e a variância de cada feature (uma "nuvem" em forma
de sino por classe) e a frequência da classe. Para classificar, usa a regra de Bayes:
a classe mais provável dado o ponto. É "ingénuo" porque supõe as features
independentes dentro de cada classe. Ficha didática: ``docs/algorithms/gaussian_nb.md``.
"""

from typing import Any, override

from pydantic import Field
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import MODELS, Implementation, ModelFamily, ModelInfo
from ml_lab.core.schemas.hyperparameters import Hyperparameters


class GaussianNBParams(Hyperparameters):
    """Hiperparâmetros do Naive Bayes gaussiano."""

    var_smoothing: float = Field(
        default=1e-9,
        ge=1e-12,
        le=1.0,
        title="Suavização da variância",
        description=(
            "Uma fração da maior variância somada a todas, para nenhuma ser zero (uma feature "
            "constante numa classe daria probabilidades de 0 ou 1)."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [1e-10, 0.1]}},
    )


@MODELS.register("gaussian_nb")
class GaussianNB(SklearnModel[GaussianNBParams]):
    """Naive Bayes gaussiano do scikit-learn."""

    info = ModelInfo(
        name="gaussian_nb",
        display_name="Naive Bayes gaussiano",
        summary=(
            "Aprende a média e a variância de cada feature em cada classe e escolhe a classe "
            "mais provável: muito rápido, mesmo com poucos dados."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.PROBABILISTIC,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=True,
        hyperparameters=GaussianNBParams,
        doc="gaussian_nb.md",
    )

    @override
    def build_estimator(self) -> SklearnGaussianNB:
        return SklearnGaussianNB(var_smoothing=self.hyperparameters.var_smoothing)

    @override
    def state(self) -> dict[str, Any]:
        estimator = self.estimator_
        return {
            "classes": estimator.classes_.tolist(),
            "priors": estimator.class_prior_.tolist(),
            "centers": estimator.theta_.tolist(),
            "variances": estimator.var_.tolist(),
        }
