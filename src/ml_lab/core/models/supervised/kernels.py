"""O que o SVM e o SVR partilham: o kernel e os seus parâmetros."""

from typing import Literal

from pydantic import Field

from ml_lab.core.schemas.hyperparameters import Hyperparameters


class KernelParams(Hyperparameters):
    """Hiperparâmetros comuns aos métodos de kernel."""

    C: float = Field(
        default=1.0,
        ge=0.001,
        le=1_000,
        title="C (cost of errors)",
        description=(
            "How much each point on the wrong side of the margin costs. Large C: a narrow "
            "margin that tries to get everything right (overfitting); small C: a wide, "
            "tolerant margin."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.1, 100]}},
    )
    kernel: Literal["rbf", "linear", "poly"] = Field(
        default="rbf",
        title="Kernel",
        description=(
            "The shape of the boundary: linear (a straight line), RBF (curves, around each "
            "point) or polynomial (of degree 3)."
        ),
        # O polinomial fica fora da procura: raramente ganha ao RBF, e com C grande demora muito.
        json_schema_extra={
            "x-labels": {"rbf": "RBF (Gaussian)", "linear": "Linear", "poly": "Polynomial"},
            "x-search": {"values": ["rbf", "linear"]},
        },
    )
    gamma: float = Field(
        default=0.5,
        ge=0.001,
        le=100,
        title="Gamma (reach of each point)",
        description=(
            "In the RBF and polynomial kernels, how far each point's influence reaches. Large "
            "gamma: each point makes its own island (overfitting); small: a smooth boundary."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.01, 2]}},
    )
