"""Engenharia de features: polinomiais, datas e intervalos.

Criar features é dar ao modelo informação numa forma que ele consiga usar:

- **Datas**: um modelo não sabe ler ``2024-12-25``, mas sabe usar o ano, o mês, o dia
  e o dia da semana;
- **Polinomiais**: ``x²`` e ``x·y`` deixam um modelo linear desenhar curvas e
  interações (e, com graus altos, mostram o overfitting);
- **Intervalos**: dividir uma coluna em faixas (por exemplo, idades em cinco grupos)
  com os quantis do treino.
"""

import itertools
from collections.abc import Sequence
from typing import Any, Final, override

import numpy as np
import pandas as pd

from ml_lab.core.data.transforms.base import Transform
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.formatting import format_number

DATE_PARTS: Final = ("year", "month", "day", "day_of_week")
"""Partes extraídas de cada data. ``day_of_week`` vai de 0 (segunda) a 6 (domingo)."""

MAX_DEGREE: Final = 3


class DateFeatures(Transform):
    """Troca cada coluna de datas pelo ano, mês, dia e dia da semana (não aprende nada).

    Args:
        columns: As colunas de datas (do tipo data do pandas).
    """

    name = "date_features"
    title = "Partes das datas"

    @staticmethod
    def output_names(column: str) -> tuple[str, ...]:
        """As colunas criadas para ``column``."""
        return tuple(f"{column}_{part}" for part in DATE_PARTS)

    @override
    def describe(self) -> str:
        created = ", ".join(name for column in self.columns for name in self.output_names(column))
        return f"Datas trocadas pelas suas partes numéricas: {created}."

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        for column in self.columns:
            if not pd.api.types.is_datetime64_any_dtype(frame[column]):
                raise DataError(
                    f"A coluna {column} não é do tipo data: converte-a na limpeza "
                    "(operação convert_type com to='datetime')."
                )

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        pieces: list[pd.DataFrame | pd.Series] = []
        for name in frame.columns:
            column = str(name)
            if column not in self.columns:
                pieces.append(frame[name])
                continue
            dates = frame[name].dt
            parts = {
                f"{column}_year": dates.year,
                f"{column}_month": dates.month,
                f"{column}_day": dates.day,
                f"{column}_day_of_week": dates.day_of_week,
            }
            pieces.append(pd.DataFrame(parts, index=frame.index).astype("float64"))
        return pd.concat(pieces, axis=1) if pieces else frame


class PolynomialFeatures(Transform):
    """Junta potências e produtos das colunas, até ao grau indicado (não aprende nada).

    Com grau 2 e colunas ``a`` e ``b`` cria ``a^2``, ``a·b`` e ``b^2``. O número de
    colunas cresce depressa com o grau: usa poucas colunas.

    Args:
        columns: As colunas numéricas.
        degree: Grau máximo (2 ou 3).
    """

    name = "polynomial_features"
    title = "Features polinomiais"

    def __init__(self, columns: Sequence[str], degree: int = 2) -> None:
        super().__init__(columns)
        if not 2 <= degree <= MAX_DEGREE:
            raise InvalidParameterError(f"O grau tem de estar entre 2 e {MAX_DEGREE}.")
        self.degree = degree

    def terms(self) -> list[tuple[str, ...]]:
        """As combinações de colunas multiplicadas, do grau 2 ao grau máximo."""
        return [
            combination
            for order in range(2, self.degree + 1)
            for combination in itertools.combinations_with_replacement(self.columns, order)
        ]

    def output_names(self) -> tuple[str, ...]:
        """Os nomes das colunas criadas, pela ordem em que aparecem."""
        return tuple(_term_name(term) for term in self.terms())

    @override
    def params(self) -> dict[str, Any]:
        return {**super().params(), "degree": self.degree}

    @override
    def describe(self) -> str:
        names = self.output_names()
        shown = ", ".join(names[:6]) + ("…" if len(names) > 6 else "")
        return (
            f"Criadas {len(names)} colunas polinomiais até ao grau {self.degree} a partir de "
            f"{', '.join(self.columns)}: {shown}."
        )

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        pass

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = {column: frame[column].astype("float64") for column in self.columns}
        created = {
            _term_name(term): np.prod([values[column] for column in term], axis=0)
            for term in self.terms()
        }
        return pd.concat([frame, pd.DataFrame(created, index=frame.index)], axis=1)


class Binning(Transform):
    """Divide cada coluna em intervalos com os quantis do treino.

    Cada valor passa a ser o número do seu intervalo (0, 1, 2…). Com quantis, cada
    intervalo tem mais ou menos o mesmo número de linhas de treino.

    Args:
        columns: As colunas numéricas a dividir.
        n_bins: Número de intervalos.
    """

    name = "binning"
    title = "Intervalos"

    def __init__(self, columns: Sequence[str], n_bins: int = 5) -> None:
        super().__init__(columns)
        if not 2 <= n_bins <= 20:
            raise InvalidParameterError("O número de intervalos tem de estar entre 2 e 20.")
        self.n_bins = n_bins
        self.edges: dict[str, tuple[float, ...]] = {}

    @staticmethod
    def output_name(column: str) -> str:
        """A coluna que substitui ``column``."""
        return f"{column}_bin"

    @override
    def params(self) -> dict[str, Any]:
        return {**super().params(), "n_bins": self.n_bins}

    @override
    def describe(self) -> str:
        parts = [
            f"{column} em {len(self.edges[column]) - 1} intervalos "
            f"({'; '.join(format_number(edge) for edge in self.edges[column])})"
            for column in self.columns
        ]
        return f"Colunas divididas com os quantis do treino: {', '.join(parts)}."

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        self.edges = {}
        for column in self.columns:
            values = frame[column].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                raise DataError(f"A coluna {column} não tem valores no treino para a dividir.")
            quantiles = np.quantile(values, np.linspace(0, 1, self.n_bins + 1))
            # Valores repetidos (por exemplo, muitos zeros) dão limites iguais: ficam só os
            # limites distintos, e por isso pode haver menos intervalos do que os pedidos.
            self.edges[column] = tuple(float(edge) for edge in np.unique(quantiles))

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        pieces: list[pd.DataFrame | pd.Series] = []
        for name in frame.columns:
            column = str(name)
            if column not in self.columns:
                pieces.append(frame[name])
                continue
            inner = np.asarray(self.edges[column][1:-1])
            values = frame[name].astype("float64")
            bins = np.digitize(values.to_numpy(), inner).astype(np.float64)
            bins[values.isna().to_numpy()] = np.nan
            pieces.append(pd.Series(bins, index=frame.index, name=self.output_name(column)))
        return pd.concat(pieces, axis=1) if pieces else frame


def polynomial_names(columns: Sequence[str], degree: int) -> tuple[str, ...]:
    """Os nomes das colunas que ``PolynomialFeatures(columns, degree)`` cria."""
    return PolynomialFeatures(columns, degree).output_names() if degree >= 2 and columns else ()


def _term_name(term: tuple[str, ...]) -> str:
    counts = {column: term.count(column) for column in dict.fromkeys(term)}
    return "·".join(
        column if power == 1 else f"{column}^{power}" for column, power in counts.items()
    )
