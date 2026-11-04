"""Etapa 9 · Saída: converte o DataFrame em arrays NumPy. Aqui acaba o pandas.

Daqui para a frente, o sistema só vê matrizes ``float64``, os nomes das features e
o número de cada classe. A conversão confirma que já não há texto nem valores em
falta, e explica o que fazer se houver: um modelo não sabe lidar com nenhum dos dois.

Também converte tabelas em pré-visualizações simples (listas de valores), a única
forma de uma tabela chegar à interface sem levar o pandas consigo.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Features, Target
from ml_lab.core.foundation.errors import DataError
from ml_lab.core.foundation.types import Partition
from ml_lab.formatting import format_quantity


@dataclass(frozen=True, slots=True)
class TablePreview:
    """As primeiras linhas de uma tabela, só com valores simples (sem pandas).

    Attributes:
        columns: Os nomes das colunas.
        dtypes: O tipo de dados de cada coluna, em texto.
        rows: Os valores de cada linha mostrada; os nulos são ``None``.
        n_rows: Quantas linhas tem a tabela inteira.
    """

    columns: tuple[str, ...]
    dtypes: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    n_rows: int

    def to_records(self) -> list[dict[str, Any]]:
        """Uma lista de dicionários, um por linha, como a interface os mostra."""
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]


def preview(frame: pd.DataFrame, n_rows: int = 20) -> TablePreview:
    """As primeiras ``n_rows`` linhas de ``frame``, convertidas em valores simples."""
    head = frame.head(n_rows)
    return TablePreview(
        columns=tuple(str(name) for name in frame.columns),
        dtypes=tuple(str(dtype) for dtype in frame.dtypes),
        rows=tuple(tuple(plain_value(value) for value in row) for row in head.itertuples(False)),
        n_rows=len(frame),
    )


def plain_value(value: object) -> Any:
    """Converte um valor do pandas ou do NumPy num valor simples do Python (nulos → ``None``)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp | datetime | date):
        return value.isoformat()
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def to_matrix(frame: pd.DataFrame, partition: Partition) -> Features:
    """Converte as features de uma parte numa matriz ``float64``.

    Raises:
        DataError: Se alguma coluna ainda não for numérica ou tiver valores em falta.
    """
    not_numeric = [str(name) for name in frame.columns if not _is_number(frame[name])]
    if not_numeric:
        raise DataError(
            f"The columns {', '.join(not_numeric)} aren't numbers yet "
            f"({partition.label.lower()} data): convert them in the cleaning or let the encoding "
            "handle them."
        )
    matrix = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    missing = np.isnan(matrix).sum(axis=0)
    if missing.any():
        details = ", ".join(
            f"{frame.columns[int(i)]} ({format_quantity(int(missing[i]), 'value', 'values')})"
            for i in np.flatnonzero(missing)
        )
        raise DataError(
            f"There are still missing values in the {partition.label.lower()} data: {details}. "
            "Turn on imputation in the transformation or drop those rows in the cleaning."
        )
    if not np.isfinite(matrix).all():
        raise DataError(
            f"There are infinite values in the {partition.label.lower()} data: check the source "
            "data."
        )
    return matrix


def class_labels(target: pd.Series) -> tuple[str, ...]:
    """Os nomes das classes de um alvo, por ordem (numérica, se forem números)."""
    present = target.dropna()
    if pd.api.types.is_numeric_dtype(present) and not pd.api.types.is_bool_dtype(present):
        return tuple(_label_text(pd.Series(np.sort(present.unique()))).tolist())
    return tuple(sorted(_label_text(present).unique().tolist()))


def encode_classes(target: pd.Series, labels: tuple[str, ...]) -> NDArray[np.int64]:
    """O número da classe de cada linha: a posição do seu nome em ``labels``.

    Raises:
        DataError: Se houver valores em falta ou classes que não estão em ``labels``.
    """
    if target.isna().any():
        raise DataError("The target still has missing values: those rows must be dropped.")
    positions = {label: index for index, label in enumerate(labels)}
    codes = _label_text(target).map(positions)
    if codes.isna().any():
        unknown = sorted(set(_label_text(target)) - set(labels))
        raise DataError(f"The target has unknown classes: {', '.join(unknown)}.")
    return np.asarray(codes.to_numpy(), dtype=np.int64)


def to_target(target: pd.Series) -> NDArray[np.float64]:
    """O alvo de uma regressão, em ``float64``.

    Raises:
        DataError: Se o alvo não for numérico ou tiver valores em falta.
    """
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        raise DataError(
            f"The target {target.name} isn't numeric: regression needs numbers. Convert it in "
            "the cleaning or choose a classification task."
        )
    values = target.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isnan(values).any():
        raise DataError("The target still has missing values: those rows must be dropped.")
    return values


def target_values(target: pd.Series, labels: tuple[str, ...]) -> Target:
    """O alvo pronto para o modelo: números de classe se houver ``labels``, senão ``float64``."""
    return encode_classes(target, labels) if labels else to_target(target)


def _is_number(values: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)


def _label_text(values: pd.Series) -> pd.Series:
    """Os valores como texto; números inteiros guardados como decimais perdem o ``.0``."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        numbers = values.astype("float64")
        if bool((numbers == numbers.round()).all()):
            return numbers.astype("int64").astype(str)
    return values.astype(str)
