"""Divisão temporal: o passado treina e o futuro testa, sem baralhar.

Quando as linhas têm uma ordem no tempo (vendas por dia, leituras de um sensor),
baralhar seria batota: o modelo aprenderia com o futuro para prever o passado. Na
divisão temporal, as linhas mais antigas são o treino, as seguintes a validação e
as mais recentes o teste, tal como o modelo seria usado na prática.
"""

import numpy as np
import pandas as pd

from ml_lab.core.data.splitting.base import SPLITTERS, SplitConfig, SplitIndices, partition_sizes
from ml_lab.core.foundation.errors import ColumnNotFoundError, DataError


@SPLITTERS.register("time")
def time_split(
    frame: pd.DataFrame, config: SplitConfig, seed: int, target: str | None
) -> SplitIndices:
    """Divide pela ordem temporal: não usa a seed nem o alvo, porque não há nada aleatório.

    As posições de cada parte ficam por ordem do tempo.

    Raises:
        ColumnNotFoundError: Se a coluna do tempo não existir.
        DataError: Se a coluna do tempo tiver nulos ou não for de datas nem numérica.
    """
    n_train, n_validation, _ = partition_sizes(len(frame), config)
    order = np.arange(len(frame), dtype=np.intp)
    if config.time_column is not None:
        column = config.time_column
        if column not in frame.columns:
            raise ColumnNotFoundError(column, [str(name) for name in frame.columns])
        times = frame[column]
        if not (
            pd.api.types.is_datetime64_any_dtype(times) or pd.api.types.is_numeric_dtype(times)
        ):
            raise DataError(
                f"The column {column} is neither dates nor numbers: convert it first in the "
                "cleaning (the convert_type operation)."
            )
        if times.isna().any():
            raise DataError(
                f"The column {column} has missing values: without a date, there's no telling "
                "where the row sits in time. Drop those rows in the cleaning."
            )
        # Ordenação estável: linhas com o mesmo instante mantêm a ordem do ficheiro.
        order = np.argsort(times.to_numpy(), kind="stable").astype(np.intp)
    return SplitIndices(
        train=order[:n_train],
        validation=order[n_train : n_train + n_validation],
        test=order[n_train + n_validation :],
    )
