"""O espaço de procura de cada modelo: o que os agentes experimentam, lido do próprio modelo.

Cada hiperparâmetro que vale a pena procurar diz como, no seu ``Field``, com a extensão
``x-search`` do JSON Schema (ver ``schemas/hyperparameters.py``):

- ``{"values": [...]}``: um dos valores da lista (as opções de um ``Literal``, ou números
  escolhidos a dedo, como as profundidades de uma árvore, onde 0 quer dizer "sem limite");
- ``{"range": [mínimo, máximo]}``: um número ao acaso entre os dois. Com ``"x-scale":
  "log"``, ao acaso nos expoentes: entre 0,01 e 100, sair entre 0,01 e 0,1 é tão provável
  como sair entre 10 e 100. É o que se quer para o ``C`` ou a taxa de aprendizagem, que
  variam em ordens de grandeza.

É a **procura aleatória** (random search): em vez de uma grelha com todas as combinações,
sorteiam-se combinações. Com o mesmo número de tentativas, experimenta mais valores
diferentes de cada hiperparâmetro, e isso conta quando só alguns deles importam (Bergstra e
Bengio, 2012). Os hiperparâmetros sem ``x-search`` ficam com o valor por omissão.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.models.base import LabModel
from ml_lab.formatting import format_value

SIGNIFICANT_DIGITS: Final = 3
"""Algarismos significativos dos números sorteados: C = 3,47 lê-se melhor do que 3,4712983."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """Um hiperparâmetro que os agentes procuram.

    Attributes:
        name: O nome do hiperparâmetro.
        title: O nome a mostrar.
        values: Os valores possíveis, numa procura por lista.
        low: O mínimo, numa procura entre dois números.
        high: O máximo, numa procura entre dois números.
        log: Se o número se sorteia nos expoentes (``x-scale`` ``log``).
        integer: Se o hiperparâmetro é um número inteiro.
        labels: Os nomes a mostrar das opções de um ``Literal`` (``x-labels``).
    """

    name: str
    title: str
    values: tuple[Any, ...] = ()
    low: float | None = None
    high: float | None = None
    log: bool = False
    integer: bool = False
    labels: Mapping[str, str] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float]:
        """O mínimo e o máximo, numa procura entre dois números.

        Raises:
            ModelError: Se a dimensão for uma lista de valores.
        """
        if self.low is None or self.high is None:
            raise ModelError(f"{self.name} procura-se numa lista de valores, não entre dois.")
        return self.low, self.high

    def sample(self, rng: np.random.Generator) -> Any:
        """Um valor ao acaso."""
        if self.values:
            return self.values[int(rng.integers(len(self.values)))]
        low, high = self.bounds
        if self.integer and not self.log:
            return int(rng.integers(int(low), int(high) + 1))
        if self.log:
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = float(rng.uniform(low, high))
        if self.integer:
            return round(value)
        return float(f"{value:.{SIGNIFICANT_DIGITS}g}")

    @property
    def description(self) -> str:
        """O que se procura, em português ("C: de 0,1 a 100, em escala logarítmica")."""
        if self.values:
            shown = (self.labels.get(str(value), written_value(value)) for value in self.values)
            return f"{self.title}: {', '.join(shown)}"
        low, high = self.bounds
        scale = ", em escala logarítmica" if self.log else ""
        if self.integer:
            return f"{self.title}: de {written_value(int(low))} a {written_value(int(high))}"
        return f"{self.title}: de {written_value(low)} a {written_value(high)}{scale}"


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """O que os agentes experimentam num modelo.

    Attributes:
        model: O modelo, no registo.
        dimensions: Os hiperparâmetros procurados; os outros ficam com o valor por omissão.
    """

    model: str
    dimensions: tuple[Dimension, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Se não há nada para procurar (o modelo só se experimenta com os valores por omissão)."""
        return not self.dimensions

    @property
    def size(self) -> int | None:
        """Quantas combinações há, se todas as dimensões forem listas (senão, são infinitas)."""
        if any(not dimension.values for dimension in self.dimensions):
            return None
        return math.prod(len(dimension.values) for dimension in self.dimensions)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Uma combinação ao acaso: um valor por dimensão."""
        return {dimension.name: dimension.sample(rng) for dimension in self.dimensions}


def search_space(model: type[LabModel[Any]]) -> SearchSpace:
    """O espaço de procura de ``model``, lido do ``x-search`` dos seus hiperparâmetros.

    Raises:
        ModelError: Se algum ``x-search`` estiver mal escrito (é um erro na definição do
            modelo, que o teste de contrato apanha).
    """
    info = model.info
    dimensions = []
    for name, spec in info.hyperparameters.model_fields.items():
        extra: Mapping[str, Any] = (
            spec.json_schema_extra if isinstance(spec.json_schema_extra, Mapping) else {}
        )
        search = extra.get("x-search")
        if search is None:
            continue
        where = f"{info.name}.{name}"
        title = spec.title or name
        integer = spec.annotation is int
        if not isinstance(search, Mapping) or len(search) != 1:
            raise ModelError(f"O x-search de {where} tem de ter só «values» ou só «range».")
        if "values" in search:
            values = search["values"]
            if not isinstance(values, list | tuple) or not values:
                raise ModelError(f"O x-search de {where} tem de ter pelo menos um valor.")
            labels = extra.get("x-labels", {})
            dimensions.append(
                Dimension(name, title, values=tuple(values), integer=integer, labels=dict(labels))
            )
        elif "range" in search:
            low, high = _bounds(search["range"], where)
            log = extra.get("x-scale") == "log"
            if log and low <= 0:
                raise ModelError(f"O x-search de {where} é logarítmico: o mínimo tem de ser > 0.")
            dimensions.append(Dimension(name, title, low=low, high=high, log=log, integer=integer))
        else:
            raise ModelError(f"O x-search de {where} tem de ter «values» ou «range».")
    return SearchSpace(info.name, tuple(dimensions))


def _bounds(bounds: Any, where: str) -> tuple[float, float]:
    """O mínimo e o máximo de um ``{"range": [mínimo, máximo]}``."""
    numbers = isinstance(bounds, list | tuple) and all(
        isinstance(bound, int | float) and not isinstance(bound, bool) for bound in bounds
    )
    if not numbers or len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ModelError(f"O x-search de {where} tem de ser [mínimo, máximo], com mínimo < máximo.")
    return float(bounds[0]), float(bounds[1])


def written_value(value: Any) -> str:
    """O valor de um hiperparâmetro como se mostra: os números à portuguesa, o resto como está."""
    return format_value(value, SIGNIFICANT_DIGITS)
