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
        title="C (custo dos erros)",
        description=(
            "Quanto custa cada ponto do lado errado da margem. C grande: margem estreita, que "
            "tenta acertar em tudo (overfitting); C pequeno: margem larga e tolerante."
        ),
        json_schema_extra={"x-scale": "log"},
    )
    kernel: Literal["rbf", "linear", "poly"] = Field(
        default="rbf",
        title="Kernel",
        description=(
            "A forma da fronteira: linear (uma reta), RBF (curvas, à volta de cada ponto) ou "
            "polinomial (de grau 3)."
        ),
        json_schema_extra={
            "x-labels": {"rbf": "RBF (gaussiano)", "linear": "Linear", "poly": "Polinomial"}
        },
    )
    gamma: float = Field(
        default=0.5,
        ge=0.001,
        le=100,
        title="Gamma (alcance de cada ponto)",
        description=(
            "Nos kernels RBF e polinomial, até onde chega a influência de cada ponto. Gamma "
            "grande: cada ponto cria a sua ilha (overfitting); pequeno: fronteira suave."
        ),
        json_schema_extra={"x-scale": "log"},
    )
