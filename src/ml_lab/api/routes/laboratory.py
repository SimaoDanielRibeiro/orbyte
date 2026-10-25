"""Rotas das demonstrações do ecrã Laboratório: um controlo, e o efeito logo a seguir."""

from typing import Literal

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.core.services import KnnDemo, LearningRateDemo, OverfittingDemo, RegularizationDemo

router = ServiceRouter(prefix="/laboratory", tags=["laboratory"], responses=ERROR_RESPONSES)


@router.get("/overfitting")
def overfitting(lab: LabDep, degree: int) -> OverfittingDemo:
    """Uma regressão polinomial de grau ``degree`` num seno com ruído: o overfitting."""
    return lab.laboratory.overfitting(degree)


@router.get("/learning-rate")
def learning_rate(lab: LabDep, rate: float) -> LearningRateDemo:
    """O gradiente descendente com a taxa de aprendizagem ``rate``."""
    return lab.laboratory.learning_rate(rate)


@router.get("/knn")
def knn(lab: LabDep, k: int) -> KnnDemo:
    """A fronteira do KNN com ``k`` vizinhos."""
    return lab.laboratory.knn(k)


@router.get("/regularization")
def regularization(
    lab: LabDep, kind: Literal["ridge", "lasso"], alpha: float
) -> RegularizationDemo:
    """Os pesos da Ridge ou do Lasso para uma gama de alphas."""
    return lab.laboratory.regularization(kind, alpha)
