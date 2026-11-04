"""Mistura de gaussianas (scikit-learn), iteração a iteração do algoritmo EM.

Supõe que os dados vêm de k "nuvens" gaussianas (cada uma com a sua média, a sua forma
e o seu peso) e descobre-as com o algoritmo EM, que repete dois passos:

1. **E** (*expectation*): a probabilidade de cada ponto pertencer a cada nuvem;
2. **M** (*maximization*): as médias, as formas e os pesos que melhor explicam essas
   probabilidades.

Ao contrário do k-means, cada ponto pertence a todas as nuvens com uma probabilidade, e
as nuvens podem ser alongadas. No treino passo a passo, cada passo é uma iteração do EM: com
``warm_start`` e uma iteração de cada vez, o scikit-learn continua de onde estava, e o
resultado é o mesmo de treinar de uma vez. Ficha didática:
``docs/algorithms/gaussian_mixture.md``.
"""

from typing import Any, Literal, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

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

type CovarianceType = Literal["full", "tied", "diag", "spherical"]


class GaussianMixtureParams(Hyperparameters):
    """Hiperparâmetros da mistura de gaussianas."""

    n_components: int = Field(
        default=3,
        ge=1,
        le=20,
        title="Number of clouds (clusters)",
        description="How many Gaussians make up the mixture.",
    )
    covariance_type: CovarianceType = Field(
        default="full",
        title="Shape of the clouds",
        description=(
            "Full: each cloud with its own ellipse, tilted freely; tied: the same ellipse for "
            "all; diagonal: ellipses aligned with the axes; spherical: circles."
        ),
        json_schema_extra={
            "x-labels": {
                "full": "Full",
                "tied": "Tied",
                "diag": "Diagonal",
                "spherical": "Spherical",
            }
        },
    )
    max_iter: int = Field(
        default=100,
        ge=1,
        le=1_000,
        title="Maximum EM iterations",
        description="How many times, at most, it repeats the E and M steps.",
    )
    tol: float = Field(
        default=0.001,
        ge=1e-6,
        le=0.1,
        title="Stopping tolerance",
        description=("Stops when the mean log-likelihood almost stops rising between iterations."),
        json_schema_extra={"x-scale": "log"},
    )


def full_covariances(estimator: Any, covariance_type: CovarianceType) -> NDArray[np.float64]:
    """As covariâncias como matrizes completas ``(k, d, d)``, seja qual for a forma escolhida."""
    covariances = np.asarray(estimator.covariances_, dtype=np.float64)
    k, d = estimator.means_.shape
    if covariance_type == "tied":
        return np.repeat(covariances[None, :, :], k, axis=0)
    if covariance_type == "diag":
        return np.stack([np.diag(row) for row in covariances])
    if covariance_type == "spherical":
        return np.stack([np.eye(d) * value for value in covariances])
    return covariances


@MODELS.register("gaussian_mixture")
class GaussianMixture(SklearnModel[GaussianMixtureParams]):
    """Mistura de gaussianas do scikit-learn, com uma iteração do EM por passo."""

    info = ModelInfo(
        name="gaussian_mixture",
        display_name="Gaussian mixture",
        summary=(
            "Explains the data as a sum of ellipse-shaped clouds: each point belongs to each "
            "cluster with a probability. You see the ellipses fit themselves."
        ),
        task_types=frozenset({TaskType.CLUSTERING}),
        family=ModelFamily.PROBABILISTIC,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=True,
        hyperparameters=GaussianMixtureParams,
        doc="gaussian_mixture.md",
        step_limit="max_iter",
    )

    @override
    def build_estimator(self) -> SklearnGaussianMixture:
        params = self.hyperparameters
        return SklearnGaussianMixture(
            n_components=params.n_components,
            covariance_type=params.covariance_type,
            max_iter=params.max_iter,
            tol=params.tol,
            n_init=1,
            random_state=self.seed,
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        params = self.hyperparameters
        if params.n_components > len(X):
            raise ModelError(
                f"You asked for {params.n_components} clouds, but the training data has only "
                f"{len(X)} rows."
            )
        if not context.observed:
            self.estimator_ = self._fit_estimator(self.build_estimator(), X, None)
            return
        stepper = clone(self.build_estimator()).set_params(max_iter=1, warm_start=True)
        for iteration in range(1, params.max_iter + 1):
            last = iteration == params.max_iter
            # Cada chamada faz uma só iteração: o aviso de "não convergiu" é de propósito.
            ignore = () if last else (ConvergenceWarning,)
            self.estimator_ = self._fit_estimator(stepper, X, None, ignore=ignore)
            context.report_step(iteration, loss=-float(stepper.lower_bound_))
            if context.wants_snapshot(iteration):
                context.report_snapshot(iteration, self.snapshot_state(context.probe))
            if stepper.converged_ or context.should_stop():
                return

    @override
    def state(self) -> dict[str, Any]:
        estimator = self.estimator_
        covariances = full_covariances(estimator, self.hyperparameters.covariance_type)
        return {
            "centers": estimator.means_.tolist(),
            "covariances": covariances.tolist(),
            "weights": estimator.weights_.tolist(),
            "converged": bool(estimator.converged_),
            "log_likelihood": float(estimator.lower_bound_),
        }
