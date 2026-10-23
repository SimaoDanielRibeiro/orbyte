"""Dados sintéticos: moons, blobs, circles e regressão sintética.

Os geradores estão escritos em NumPy, com a mesma geometria dos ``make_*`` do
scikit-learn, para se ver como cada forma é construída. Cada gerador declara os
seus parâmetros (com limites e valores por omissão), que a interface transforma
em controlos, e recebe uma seed: a mesma seed dá sempre os mesmos pontos.

Para acrescentar um gerador: escreve uma função ``(rng, **parâmetros) -> Generated``
neste ficheiro e decora-a com ``@generator(...)``.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_lab.core.data.io.base import DATA_SOURCES, DataSource, LoadedTable, SourceSpec
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.random import validate_seed
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import TaskType
from ml_lab.formatting import format_number

TARGET: Final = "target"
"""Nome da coluna-alvo em todos os datasets sintéticos."""


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Um parâmetro numérico de um gerador, com os seus limites.

    Attributes:
        name: Nome do argumento da função.
        label: Nome a mostrar na interface.
        kind: ``"integer"`` ou ``"number"``, como no JSON Schema.
        default: Valor por omissão.
        minimum: Menor valor aceite.
        maximum: Maior valor aceite.
        help: Explicação curta do efeito do parâmetro.
    """

    name: str
    label: str
    kind: Literal["integer", "number"]
    default: float
    minimum: float
    maximum: float
    help: str = ""

    def json_schema(self) -> dict[str, Any]:
        """O parâmetro em JSON Schema, o formato que a interface usa para gerar controlos."""
        cast = int if self.kind == "integer" else float
        return {
            "type": self.kind,
            "title": self.label,
            "description": self.help,
            "default": cast(self.default),
            "minimum": cast(self.minimum),
            "maximum": cast(self.maximum),
        }

    def validate(self, value: object) -> int | float:
        """Confirma que ``value`` é um número dentro dos limites.

        Raises:
            InvalidParameterError: Se não for um número, não for inteiro quando devia, ou
                estiver fora dos limites.
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvalidParameterError(f"'{self.label}' tem de ser um número.")
        if self.kind == "integer" and not float(value).is_integer():
            raise InvalidParameterError(f"'{self.label}' tem de ser um número inteiro.")
        if not self.minimum <= value <= self.maximum:
            decimals = 0 if self.kind == "integer" else 2
            raise InvalidParameterError(
                f"'{self.label}' tem de estar entre {format_number(self.minimum, decimals)} e "
                f"{format_number(self.maximum, decimals)}; recebi {format_number(value, 2)}."
            )
        return int(value) if self.kind == "integer" else float(value)


@dataclass(frozen=True, slots=True, eq=False)
class Generated:
    """O que um gerador devolve: a tabela e, se fizer sentido, a regra que a gerou."""

    frame: pd.DataFrame
    details: str = ""


type GeneratorFunction = Callable[..., Generated]
"""Função ``(rng, **parâmetros) -> Generated``."""


@dataclass(frozen=True, slots=True)
class GeneratorInfo:
    """Ficha de um gerador de dados sintéticos.

    Attributes:
        name: Identificador, igual ao nome no registo.
        title: Nome a mostrar.
        description: Que forma os dados têm e para que servem.
        task_type: Tipo de tarefa para que os dados foram pensados.
        parameters: Parâmetros que o utilizador pode mudar.
        function: A função que gera os dados.
    """

    name: str
    title: str
    description: str
    task_type: TaskType
    parameters: tuple[ParameterSpec, ...]
    function: GeneratorFunction

    def parameters_schema(self) -> dict[str, Any]:
        """Os parâmetros em JSON Schema (um objeto com uma propriedade por parâmetro)."""
        return {
            "type": "object",
            "properties": {p.name: p.json_schema() for p in self.parameters},
            "required": [p.name for p in self.parameters],
        }

    def resolve(self, values: Mapping[str, object]) -> dict[str, int | float]:
        """Junta os valores dados aos de omissão e valida-os.

        Raises:
            InvalidParameterError: Se houver parâmetros desconhecidos ou valores inválidos.
        """
        known = {p.name for p in self.parameters}
        unknown = sorted(set(values) - known)
        if unknown:
            raise InvalidParameterError(
                f"O gerador '{self.name}' não tem os parâmetros {', '.join(unknown)}. "
                f"Aceita: {', '.join(sorted(known))}."
            )
        return {p.name: p.validate(values.get(p.name, p.default)) for p in self.parameters}

    def generate(self, values: Mapping[str, object], seed: int) -> Generated:
        """Gera os dados com os parâmetros ``values`` e a ``seed`` dada."""
        rng = np.random.default_rng(validate_seed(seed))
        return self.function(rng, **self.resolve(values))


GENERATORS: Registry[GeneratorInfo] = Registry(
    "generators", task_types=lambda info: {info.task_type}
)
"""Geradores de dados sintéticos, pelo nome."""


def generator(
    name: str,
    *,
    title: str,
    description: str,
    task_type: TaskType,
    parameters: tuple[ParameterSpec, ...],
) -> Callable[[GeneratorFunction], GeneratorFunction]:
    """Decorador que regista uma função como gerador de dados sintéticos."""

    def decorator(function: GeneratorFunction) -> GeneratorFunction:
        GENERATORS.add(
            name, GeneratorInfo(name, title, description, task_type, parameters, function)
        )
        return function

    return decorator


@DATA_SOURCES.register("synthetic")
@dataclass(frozen=True, slots=True)
class SyntheticSource(DataSource):
    """Dados gerados por um dos ``GENERATORS``.

    Attributes:
        generator: Nome do gerador.
        seed: Seed do gerador; a mesma seed dá os mesmos dados.
        parameters: Valores dos parâmetros; os que faltam usam o valor por omissão.
    """

    generator: str
    seed: int
    parameters: Mapping[str, float] | None = None

    @override
    def load(self) -> LoadedTable:
        info = GENERATORS.get(self.generator)
        generated = info.generate(self.parameters or {}, self.seed)
        description = f"{info.description} Seed: {self.seed}."
        if generated.details:
            description = f"{description} {generated.details}"
        return LoadedTable(
            name=f"{info.title} (sintético)",
            frame=generated.frame,
            spec=self.spec(),
            description=description,
            suggested_target=TARGET,
        )

    @override
    def spec(self) -> SourceSpec:
        # Guarda todos os parâmetros, incluindo os de omissão: se a omissão mudar um dia,
        # a experiência continua a dar os mesmos dados.
        info = GENERATORS.get(self.generator)
        resolved = info.resolve(self.parameters or {})
        return SourceSpec(
            "synthetic", {"generator": self.generator, "seed": self.seed, "parameters": resolved}
        )


# --- geradores --------------------------------------------------------------------------

_N_SAMPLES = ParameterSpec(
    "n_samples", "Número de pontos", "integer", 200, 20, 5_000, "Quantas linhas gerar."
)


def _table(
    points: NDArray[np.float64], target: NDArray[np.generic], rng: np.random.Generator
) -> pd.DataFrame:
    """Baralha as linhas e monta a tabela ``x1, x2, …, target``."""
    order = rng.permutation(len(points))
    columns = {f"x{index + 1}": points[order, index] for index in range(points.shape[1])}
    return pd.DataFrame({**columns, TARGET: target[order]})


@generator(
    "moons",
    title="Luas",
    description=(
        "Duas meias-luas encaixadas uma na outra. Não se separam com uma reta, por isso mostram "
        "o que um modelo não linear consegue e um linear não."
    ),
    task_type=TaskType.BINARY_CLASSIFICATION,
    parameters=(
        _N_SAMPLES,
        ParameterSpec(
            "noise", "Ruído", "number", 0.2, 0.0, 1.0, "Desvio padrão do ruído somado a cada ponto."
        ),
    ),
)
def moons(rng: np.random.Generator, *, n_samples: int, noise: float) -> Generated:
    """Meia-lua de cima (classe 0) e meia-lua de baixo, deslocada (classe 1)."""
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    outer = np.linspace(0, math.pi, n_outer)
    inner = np.linspace(0, math.pi, n_inner)
    points = np.column_stack(
        [
            np.concatenate([np.cos(outer), 1 - np.cos(inner)]),
            np.concatenate([np.sin(outer), 1 - np.sin(inner) - 0.5]),
        ]
    )
    points += rng.normal(scale=noise, size=points.shape)
    target = np.repeat([0, 1], [n_outer, n_inner])
    return Generated(_table(points, target, rng))


@generator(
    "circles",
    title="Círculos",
    description=(
        "Um círculo dentro de outro: o caso clássico em que nenhuma fronteira linear funciona."
    ),
    task_type=TaskType.BINARY_CLASSIFICATION,
    parameters=(
        _N_SAMPLES,
        ParameterSpec("noise", "Ruído", "number", 0.1, 0.0, 0.5, "Desvio padrão do ruído."),
        ParameterSpec(
            "factor",
            "Raio interior",
            "number",
            0.5,
            0.1,
            0.9,
            "Raio do círculo interior, em proporção do exterior.",
        ),
    ),
)
def circles(rng: np.random.Generator, *, n_samples: int, noise: float, factor: float) -> Generated:
    """Círculo exterior de raio 1 (classe 0) e interior de raio ``factor`` (classe 1)."""
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    outer = np.linspace(0, 2 * math.pi, n_outer, endpoint=False)
    inner = np.linspace(0, 2 * math.pi, n_inner, endpoint=False)
    points = np.column_stack(
        [
            np.concatenate([np.cos(outer), factor * np.cos(inner)]),
            np.concatenate([np.sin(outer), factor * np.sin(inner)]),
        ]
    )
    points += rng.normal(scale=noise, size=points.shape)
    target = np.repeat([0, 1], [n_outer, n_inner])
    return Generated(_table(points, target, rng))


@generator(
    "blobs",
    title="Nuvens (blobs)",
    description=(
        "Nuvens de pontos à volta de centros escolhidos ao acaso. Servem para clustering (sem "
        "alvo) e para classificação multiclasse (com a nuvem como alvo)."
    ),
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    parameters=(
        ParameterSpec("n_samples", "Número de pontos", "integer", 300, 20, 5_000),
        ParameterSpec("centers", "Número de nuvens", "integer", 3, 2, 8),
        ParameterSpec(
            "cluster_std",
            "Dispersão",
            "number",
            1.0,
            0.1,
            5.0,
            "Desvio padrão de cada nuvem: valores altos fazem as nuvens sobrepor-se.",
        ),
        ParameterSpec("n_features", "Número de features", "integer", 2, 2, 10),
    ),
)
def blobs(
    rng: np.random.Generator,
    *,
    n_samples: int,
    centers: int,
    cluster_std: float,
    n_features: int,
) -> Generated:
    """Cada nuvem é uma normal à volta de um centro uniforme em [-10, 10]."""
    center_points = rng.uniform(-10, 10, size=(centers, n_features))
    counts = np.full(centers, n_samples // centers)
    counts[: n_samples % centers] += 1
    points = np.vstack(
        [
            rng.normal(loc=center, scale=cluster_std, size=(count, n_features))
            for center, count in zip(center_points, counts, strict=True)
        ]
    )
    target = np.repeat(np.arange(centers), counts)
    return Generated(_table(points, target, rng))


@generator(
    "regression",
    title="Regressão linear",
    description=(
        "Um alvo que é uma combinação linear das features mais ruído. Um bom modelo de "
        "regressão deve reencontrar os coeficientes usados."
    ),
    task_type=TaskType.REGRESSION,
    parameters=(
        _N_SAMPLES,
        ParameterSpec("n_features", "Número de features", "integer", 1, 1, 10),
        ParameterSpec(
            "noise", "Ruído", "number", 1.0, 0.0, 20.0, "Desvio padrão do ruído somado ao alvo."
        ),
    ),
)
def regression(
    rng: np.random.Generator, *, n_samples: int, n_features: int, noise: float
) -> Generated:
    """``target = x · coeficientes + ordenada + ruído``, com features normais padrão."""
    points = rng.normal(size=(n_samples, n_features))
    # Coeficientes com uma casa decimal, para se lerem facilmente na descrição.
    coefficients = np.round(rng.uniform(-5, 5, size=n_features), 1)
    intercept = float(np.round(rng.uniform(-5, 5), 1))
    target = points @ coefficients + intercept + rng.normal(scale=noise, size=n_samples)
    terms = " + ".join(
        f"{format_number(c, 1)}·x{index + 1}" for index, c in enumerate(coefficients)
    )
    details = (
        f"Regra usada: target = {terms} + {format_number(intercept, 1)} + ruído "
        f"(desvio padrão {format_number(noise, 1)})."
    )
    return Generated(_table(points, target, rng), details)
