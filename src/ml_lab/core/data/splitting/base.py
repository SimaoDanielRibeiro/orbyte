"""Divisão em treino, validação e teste: configuração, resultado e registo das estratégias.

O modelo aprende com o treino e é avaliado no teste, em linhas que nunca viu: é a
única forma honesta de saber se aprendeu a generalizar ou se só decorou. A
validação (opcional) serve para afinar escolhas sem tocar no teste.

Toda a divisão aleatória usa a seed recebida, por isso a mesma seed dá sempre a
mesma divisão (ADR 0006). As estratégias registam-se em ``SPLITTERS``; para
acrescentar uma, cria um ficheiro nesta pasta com uma função decorada com
``@SPLITTERS.register("nome")``.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_lab.core.data.lineage import LineageStep, Stage
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import Partition
from ml_lab.formatting import format_percent, format_quantity

type Positions = NDArray[np.intp]
"""Posições das linhas (0, 1, 2…) na tabela dividida."""


class SplitStrategy(StrEnum):
    """Como escolher que linhas vão para cada parte."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME = "time"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface."""
        return _STRATEGY_LABELS[self]


_STRATEGY_LABELS: Final[Mapping[SplitStrategy, str]] = MappingProxyType(
    {
        SplitStrategy.RANDOM: "Random",
        SplitStrategy.STRATIFIED: "Stratified",
        SplitStrategy.TIME: "Time-based",
    }
)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Configuração da divisão.

    Attributes:
        strategy: Aleatória, estratificada (mantém a proporção de cada classe) ou temporal
            (o passado treina, o futuro testa).
        test_size: Proporção das linhas que vai para o teste.
        validation_size: Proporção que vai para a validação (0 para não haver validação).
        time_column: Na divisão temporal, a coluna com o tempo; sem ela, usa-se a ordem
            das linhas.
    """

    strategy: SplitStrategy = SplitStrategy.RANDOM
    test_size: float = 0.2
    validation_size: float = 0.0
    time_column: str | None = None

    def __post_init__(self) -> None:
        # Aceita o nome da estratégia em texto ("random"), como chega de JSON.
        object.__setattr__(self, "strategy", SplitStrategy(self.strategy))
        if not 0 < self.test_size < 1:
            raise InvalidParameterError(
                f"The test share must be between 0 and 1 (0.2, for example); got {self.test_size}."
            )
        if not 0 <= self.validation_size < 1:
            raise InvalidParameterError(
                f"The validation share must be between 0 and 1; got {self.validation_size}."
            )
        if self.test_size + self.validation_size >= 1:
            raise InvalidParameterError(
                "Together, the test and the validation must leave rows for the training: the two "
                "shares must add up to less than 1."
            )


@dataclass(frozen=True, slots=True, eq=False)
class SplitIndices:
    """As posições das linhas de cada parte. As partes não se sobrepõem.

    Attributes:
        train: Posições das linhas de treino.
        validation: Posições das linhas de validação (vazio se não houver).
        test: Posições das linhas de teste.
    """

    train: Positions
    validation: Positions
    test: Positions

    def of(self, partition: Partition) -> Positions:
        """As posições de uma parte."""
        return {
            Partition.TRAIN: self.train,
            Partition.VALIDATION: self.validation,
            Partition.TEST: self.test,
        }[partition]

    @property
    def sizes(self) -> dict[Partition, int]:
        """Número de linhas de cada parte (sem a validação, se estiver vazia)."""
        sizes = {Partition.TRAIN: len(self.train)}
        if len(self.validation):
            sizes[Partition.VALIDATION] = len(self.validation)
        sizes[Partition.TEST] = len(self.test)
        return sizes


type Splitter = Callable[[pd.DataFrame, SplitConfig, int, str | None], SplitIndices]
"""Função ``(tabela, configuração, seed, alvo) -> SplitIndices``."""

SPLITTERS: Registry[Splitter] = Registry("splitters", discover="ml_lab.core.data.splitting")
"""Estratégias de divisão, pelo nome (o valor de ``SplitStrategy``)."""


def partition_sizes(n_rows: int, config: SplitConfig) -> tuple[int, int, int]:
    """Quantas linhas vão para treino, validação e teste.

    Arredonda o teste e a validação para cima, como o scikit-learn, por isso 891
    linhas com 20% de teste dão 712 de treino e 179 de teste.

    Raises:
        DataError: Se não sobrarem linhas para alguma das partes.
    """
    n_test = math.ceil(n_rows * config.test_size)
    n_validation = math.ceil(n_rows * config.validation_size)
    n_train = n_rows - n_test - n_validation
    if n_train < 1:
        raise DataError(
            f"With {format_quantity(n_rows, 'row', 'rows')} there aren't enough rows for "
            "training, validation and test: use smaller shares or more data."
        )
    return n_train, n_validation, n_test


def split_rows(
    frame: pd.DataFrame,
    config: SplitConfig,
    *,
    seed: int,
    target: str | None = None,
    root_seed: int | None = None,
) -> tuple[SplitIndices, LineageStep]:
    """Divide as linhas de ``frame`` segundo ``config``.

    Args:
        frame: A tabela, já limpa.
        config: A estratégia e as proporções.
        seed: Seed da divisão (normalmente ``seeds.seed_for("split")``).
        target: A coluna-alvo, necessária na divisão estratificada.
        root_seed: A seed raiz da experiência de onde ``seed`` foi derivada, para a linhagem.

    Returns:
        As posições de cada parte e o passo de linhagem.
    """
    indices = SPLITTERS.get(config.strategy.value)(frame, config, seed, target)
    n_rows = len(frame)
    parts = ", ".join(
        f"{partition.label.lower()} {format_quantity(size, 'row', 'rows')} "
        f"({format_percent(size / n_rows)})"
        for partition, size in indices.sizes.items()
    )
    if config.strategy is SplitStrategy.TIME:
        order = f"by {config.time_column}" if config.time_column else "in the order of the rows"
        description = f"Time-based split {order}, without shuffling: {parts}."
    else:
        origin = (
            f", derived from the experiment's seed {root_seed}" if root_seed is not None else ""
        )
        description = f"{config.strategy.label} split with the seed {seed}{origin}: {parts}."
    params: dict[str, object] = {
        "strategy": config.strategy.value,
        "test_size": config.test_size,
        "validation_size": config.validation_size,
        "seed": seed,
    }
    if config.time_column:
        params["time_column"] = config.time_column
    step = LineageStep(
        stage=Stage.SPLIT,
        operation=f"{config.strategy.value}_split",
        description=description,
        params=params,
        rows_before=n_rows,
        rows_after=len(indices.train),
    )
    return indices, step


def sorted_positions(values: NDArray[np.integer]) -> Positions:
    """As posições por ordem crescente, no tipo de índice do NumPy."""
    return np.sort(values).astype(np.intp)
