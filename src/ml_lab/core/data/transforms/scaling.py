"""Escala das features numéricas.

Colunas com escalas muito diferentes (a prolina do Wine chega aos milhares, a
matiz não passa de 2) confundem os modelos que medem distâncias (KNN, SVM) ou que
aprendem por gradiente: a coluna com números maiores domina. Escalar põe todas na
mesma ordem de grandeza.

- **Padronização**: subtrai a média e divide pelo desvio padrão (média 0, desvio 1).
- **Min-max**: põe o mínimo em 0 e o máximo em 1.

A média, o desvio, o mínimo e o máximo são aprendidos no treino. Por isso, no
teste, os valores padronizados não têm exatamente média 0, e os de min-max podem
sair do intervalo [0, 1]: é o esperado.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Final, override

import pandas as pd

from ml_lab.core.data.transforms.base import Transform
from ml_lab.formatting import format_number

_EXAMPLES: Final = 3


class ScalingMethod(StrEnum):
    """Como escalar as colunas numéricas."""

    STANDARD = "standard"
    MINMAX = "minmax"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _LABELS[self]


_LABELS: Final[Mapping[ScalingMethod, str]] = MappingProxyType(
    {
        ScalingMethod.STANDARD: "Padronização (média 0, desvio 1)",
        ScalingMethod.MINMAX: "Min-max (entre 0 e 1)",
    }
)


class StandardScaler(Transform):
    """``(x - média) / desvio``, com a média e o desvio padrão do treino.

    O desvio é o da população (divide por n, não por n - 1), como no scikit-learn.
    Uma coluna constante no treino tem desvio 0: fica só centrada, sem dividir.

    Args:
        columns: As colunas numéricas a escalar.
    """

    name = "standard_scaler"
    title = "Padronização"

    def __init__(self, columns: Sequence[str]) -> None:
        super().__init__(columns)
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}

    @override
    def describe(self) -> str:
        examples = "; ".join(
            f"{column}: média {format_number(self.means[column])}, desvio "
            f"{format_number(self.scales[column])}"
            for column in self.columns[:_EXAMPLES]
        )
        more = "…" if len(self.columns) > _EXAMPLES else ""
        return (
            f"{len(self.columns)} colunas padronizadas com a média e o desvio do treino "
            f"({examples}{more})."
        )

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        values = frame[list(self.columns)].astype("float64")
        self.means = {str(c): float(v) for c, v in values.mean().items()}
        stds = values.std(ddof=0)
        self.scales = {str(c): float(v) if v > 0 else 1.0 for c, v in stds.items()}

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(
            **{
                column: (frame[column].astype("float64") - self.means[column]) / self.scales[column]
                for column in self.columns
            }
        )


class MinMaxScaler(Transform):
    """``(x - mínimo) / (máximo - mínimo)``, com o mínimo e o máximo do treino.

    Args:
        columns: As colunas numéricas a escalar.
    """

    name = "minmax_scaler"
    title = "Min-max"

    def __init__(self, columns: Sequence[str]) -> None:
        super().__init__(columns)
        self.minimums: dict[str, float] = {}
        self.ranges: dict[str, float] = {}

    @override
    def describe(self) -> str:
        examples = "; ".join(
            f"{column}: de {format_number(self.minimums[column])} a "
            f"{format_number(self.minimums[column] + self.ranges[column])}"
            for column in self.columns[:_EXAMPLES]
        )
        more = "…" if len(self.columns) > _EXAMPLES else ""
        return (
            f"{len(self.columns)} colunas levadas para [0, 1] com o mínimo e o máximo do treino "
            f"({examples}{more})."
        )

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        values = frame[list(self.columns)].astype("float64")
        self.minimums = {str(c): float(v) for c, v in values.min().items()}
        spans = values.max() - values.min()
        self.ranges = {str(c): float(v) if v > 0 else 1.0 for c, v in spans.items()}

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(
            **{
                column: (frame[column].astype("float64") - self.minimums[column])
                / self.ranges[column]
                for column in self.columns
            }
        )
