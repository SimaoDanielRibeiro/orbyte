"""Imputação de valores em falta.

Preenche os nulos com um valor aprendido no treino: a média ou a mediana nas
colunas numéricas, o valor mais frequente nas categóricas. A mediana é a escolha
por omissão porque não se deixa arrastar por valores extremos (no Titanic, a média
das tarifas é o dobro da mediana).

Imputar não recupera a informação que falta: só deixa o modelo usar as outras
colunas dessas linhas. Com muitos nulos, a coluna fica cheia de valores inventados.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, override

import pandas as pd

from ml_lab.core.data.transforms.base import Transform
from ml_lab.formatting import format_number, format_quantity


class ImputationStrategy(StrEnum):
    """Como escolher o valor que preenche os nulos."""

    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _LABELS[self]


_LABELS: Final[Mapping[ImputationStrategy, str]] = MappingProxyType(
    {
        ImputationStrategy.MEAN: "Média",
        ImputationStrategy.MEDIAN: "Mediana",
        ImputationStrategy.MOST_FREQUENT: "Valor mais frequente",
    }
)

_WITH_ARTICLE: Final[Mapping[ImputationStrategy, str]] = MappingProxyType(
    {
        ImputationStrategy.MEAN: "a média",
        ImputationStrategy.MEDIAN: "a mediana",
        ImputationStrategy.MOST_FREQUENT: "o valor mais frequente",
    }
)

NUMERIC_STRATEGIES: Final = (ImputationStrategy.MEDIAN, ImputationStrategy.MEAN)
CATEGORICAL_STRATEGIES: Final = (ImputationStrategy.MOST_FREQUENT,)


class Imputer(Transform):
    """Preenche os nulos de cada coluna com um valor aprendido no treino.

    Args:
        columns: As colunas a preencher.
        strategy: Média, mediana (só colunas numéricas) ou valor mais frequente.
    """

    name = "impute"
    title = "Imputação"

    def __init__(self, columns: Sequence[str], strategy: ImputationStrategy) -> None:
        super().__init__(columns)
        self.strategy = ImputationStrategy(strategy)
        self.fill_values: dict[str, Any] = {}
        self._missing_in_train: dict[str, int] = {}

    @override
    def params(self) -> dict[str, Any]:
        return {**super().params(), "strategy": self.strategy.value}

    @override
    def describe(self) -> str:
        how = _WITH_ARTICLE[self.strategy]
        filled = [
            f"{column} ({format_quantity(self._missing_in_train[column], 'nulo', 'nulos')}) com "
            f"{_show(self.fill_values[column])}"
            for column in self.columns
            if self._missing_in_train[column]
        ]
        others = len(self.columns) - len(filled)
        if not filled:
            return (
                f"Nenhuma das {format_quantity(others, 'coluna', 'colunas')} tinha nulos no "
                f"treino; ficou guardada {how} de cada uma, para preencher nulos que apareçam "
                "na validação, no teste ou em dados novos."
            )
        text = f"Nulos do treino preenchidos com {how} do treino: {'; '.join(filled)}."
        if others:
            text += (
                f" {format_quantity(others, 'outra coluna não tinha', 'outras colunas não tinham')}"
                " nulos, mas fica com o valor guardado para dados novos."
            )
        return f"{text} A validação e o teste usam estes mesmos valores."

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        self.fill_values = {}
        self._missing_in_train = {}
        for column in self.columns:
            values = frame[column]
            self._missing_in_train[column] = int(values.isna().sum())
            self.fill_values[column] = self._learn(values)

    def _learn(self, values: pd.Series) -> Any:
        present = values.dropna()
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if present.empty:
            # Uma coluna sem nenhum valor no treino não tem nada para ensinar.
            return 0.0 if numeric else "em falta"
        if self.strategy is ImputationStrategy.MEAN and numeric:
            return float(present.mean())
        if self.strategy is ImputationStrategy.MEDIAN and numeric:
            return float(present.median())
        # mode() devolve os empates por ordem crescente: fica o primeiro, sempre o mesmo.
        return present.mode().iloc[0]

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.fillna(self.fill_values)


def _show(value: Any) -> str:
    if isinstance(value, float):
        return format_number(value)
    return repr(value) if isinstance(value, str) else str(value)
