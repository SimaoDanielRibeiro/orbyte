"""Codificação de variáveis categóricas.

Os modelos só fazem contas com números. A codificação *one-hot* troca uma coluna
categórica por uma coluna 0/1 por categoria: ``Embarked`` passa a ``Embarked_C``,
``Embarked_Q`` e ``Embarked_S``. Não se usa 1, 2, 3, porque isso inventaria uma
ordem (S > Q > C) que não existe.

As categorias são aprendidas no treino. Uma categoria que só apareça no teste
fica com zeros em todas as colunas: o modelo nunca a viu, por isso não lhe pode
dar significado.
"""

from collections.abc import Mapping, Sequence
from typing import Any, override

import numpy as np
import pandas as pd

from ml_lab.core.data.profiling.type_inference import BOOLEAN_WORDS
from ml_lab.core.data.transforms.base import Transform
from ml_lab.formatting import format_quantity


class OneHotEncoder(Transform):
    """Uma coluna 0/1 por categoria vista no treino.

    Args:
        columns: As colunas categóricas a codificar.
    """

    name = "one_hot"
    title = "Codificação one-hot"

    def __init__(self, columns: Sequence[str]) -> None:
        super().__init__(columns)
        self.categories: dict[str, tuple[str, ...]] = {}

    @override
    def describe(self) -> str:
        parts = [
            f"{column} → {format_quantity(len(self.categories[column]), 'coluna', 'colunas')} "
            f"({', '.join(self.output_names(column))})"
            for column in self.columns
        ]
        return (
            f"Categorias aprendidas no treino e trocadas por colunas 0/1: {'; '.join(parts)}. "
            "Uma categoria que só apareça na validação ou no teste fica com zeros."
        )

    def output_names(self, column: str) -> tuple[str, ...]:
        """Os nomes das colunas criadas para ``column`` (depois do ``fit``)."""
        return tuple(f"{column}_{category}" for category in self.categories[column])

    def unknown_counts(self, frame: pd.DataFrame) -> dict[str, int]:
        """Quantos valores de cada coluna não eram categorias conhecidas no treino."""
        counts: dict[str, int] = {}
        for column in self.columns:
            text = _as_text(frame[column])
            unknown = int((text.notna() & ~text.isin(self.categories[column])).sum())
            if unknown:
                counts[column] = unknown
        return counts

    @override
    def _learned(self) -> dict[str, Any]:
        return {"categories": {column: list(values) for column, values in self.categories.items()}}

    @override
    def _restore(self, learned: Mapping[str, Any]) -> None:
        self.categories = {
            str(column): tuple(str(value) for value in values)
            for column, values in learned["categories"].items()
        }

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        self.categories = {
            column: tuple(sorted(_as_text(frame[column]).dropna().unique().tolist()))
            for column in self.columns
        }

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        pieces: list[pd.DataFrame | pd.Series] = []
        for name in frame.columns:
            column = str(name)
            if column not in self.columns:
                pieces.append(frame[name])
                continue
            text = _as_text(frame[name])
            encoded = {
                f"{column}_{category}": (text == category).to_numpy(dtype=np.float64)
                for category in self.categories[column]
            }
            pieces.append(pd.DataFrame(encoded, index=frame.index))
        if not pieces:
            return frame
        return pd.concat(pieces, axis=1)


class BooleanEncoder(Transform):
    """Converte colunas booleanas em 0 e 1 (não aprende nada).

    Aceita ``True``/``False``, 0/1 e palavras como ``sim``/``não``. Os nulos
    continuam nulos (a imputação trata deles antes).

    Args:
        columns: As colunas booleanas.
    """

    name = "boolean_to_number"
    title = "Booleanos para 0/1"

    @override
    def describe(self) -> str:
        columns = ", ".join(self.columns)
        return f"Colunas booleanas convertidas em 1 (verdadeiro) e 0 (falso): {columns}."

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        pass

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{column: to_zero_one(frame[column]) for column in self.columns})


def to_zero_one(values: pd.Series) -> pd.Series:
    """Converte uma coluna booleana (``bool``, 0/1 ou palavras) em ``float64`` 0/1."""
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        # Float64 (com F maiúsculo) aceita os nulos do tipo "boolean"; depois viram NaN.
        numbers = values.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
        return pd.Series(numbers, index=values.index, name=values.name)
    words: dict[str, Any] = {word: float(flag) for word, flag in BOOLEAN_WORDS.items()}
    return _as_text(values).str.lower().map(words).astype("float64")


def _as_text(values: pd.Series) -> pd.Series:
    """Os valores como texto, para comparar categorias; os nulos continuam nulos."""
    return values.astype("str").where(values.notna())
