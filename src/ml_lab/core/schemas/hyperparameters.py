"""Base dos hiperparâmetros dos modelos (Pydantic).

Cada modelo declara os seus hiperparâmetros numa subclasse de ``Hyperparameters``,
com tipos, limites, valores por omissão e textos em português::

    class KnnParams(Hyperparameters):
        n_neighbors: int = Field(5, ge=1, le=100, title="Número de vizinhos")

Daqui sai tudo o resto, sem código específico por modelo: a validação dos valores
(com mensagens em português), o JSON para guardar a experiência e o JSON Schema de
onde as duas interfaces geram o formulário.

Duas extensões ao JSON Schema, que os formulários entendem: ``"x-scale": "log"``
para números que variam em ordens de grandeza, e ``"x-labels"`` com os nomes a
mostrar para as opções de um ``Literal``.
"""

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from ml_lab.core.foundation.errors import HyperparameterError
from ml_lab.core.schemas.validation import describe_errors


class Hyperparameters(BaseModel):
    """Base dos hiperparâmetros: imutáveis e sem campos desconhecidos."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def parse(cls, values: Mapping[str, Any], *, model: str = "modelo") -> Self:
        """Valida ``values``; os que faltam ficam com o valor por omissão.

        Raises:
            HyperparameterError: Com uma frase em português por cada valor recusado.
        """
        try:
            return cls.model_validate(dict(values))
        except ValidationError as error:
            raise HyperparameterError(model, describe_errors(error, cls.titles())) from error

    @classmethod
    def titles(cls) -> dict[str, str]:
        """O título de cada hiperparâmetro (o nome a mostrar)."""
        return {name: field.title or name for name, field in cls.model_fields.items()}

    @classmethod
    def form_schema(cls) -> dict[str, Any]:
        """O JSON Schema de onde as interfaces geram o formulário."""
        return cls.model_json_schema()
