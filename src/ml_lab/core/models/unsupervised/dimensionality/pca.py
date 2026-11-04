"""PCA (scikit-learn).

As mesmas componentes principais da versão didática (``pca_numpy.py``), calculadas com a
decomposição em valores singulares (SVD) dos dados centrados, que é mais estável do que
a matriz de covariância. Ficha didática: ``docs/algorithms/pca.md``.
"""

from typing import Any, override

from pydantic import Field
from sklearn import decomposition

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
from ml_lab.core.models.unsupervised.dimensionality.pca_numpy import PCAParams, check_components
from ml_lab.core.training.context import TrainingContext


class SklearnPCAParams(PCAParams):
    """Hiperparâmetros da PCA do scikit-learn."""

    whiten: bool = Field(
        default=False,
        title="Whiten",
        description=(
            "Divide each component by its standard deviation, so they all have variance 1. "
            "Useful before other models; you lose track of which component varies most."
        ),
    )


@MODELS.register("pca")
class PCA(SklearnModel[SklearnPCAParams]):
    """PCA do scikit-learn."""

    info = ModelInfo(
        name="pca",
        display_name="PCA",
        summary=(
            "Sums up many features in a few directions, those of greatest variance: to see the "
            "data on a chart or to simplify it before another model."
        ),
        task_types=frozenset({TaskType.DIMENSIONALITY_REDUCTION}),
        family=ModelFamily.PROJECTION,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=SklearnPCAParams,
        doc="pca.md",
    )

    @override
    def build_estimator(self) -> decomposition.PCA:
        params = self.hyperparameters
        return decomposition.PCA(
            n_components=params.n_components, whiten=params.whiten, random_state=self.seed
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        check_components(self.hyperparameters.n_components, X)
        super()._fit(X, None, context)

    @override
    def state(self) -> dict[str, Any]:
        estimator = self.estimator_
        return {
            "components": estimator.components_.tolist(),
            "explained_variance": estimator.explained_variance_.tolist(),
            "explained_variance_ratio": estimator.explained_variance_ratio_.tolist(),
            "mean": estimator.mean_.tolist(),
        }
