"""Divisão aleatória.

Baralha as linhas com a seed e corta-as em teste, validação e treino. É a escolha
habitual quando as linhas são independentes umas das outras e não há classes
raras; com classes desequilibradas, a estratificada é mais segura.
"""

import numpy as np
import pandas as pd

from ml_lab.core.data.splitting.base import (
    SPLITTERS,
    SplitConfig,
    SplitIndices,
    partition_sizes,
    sorted_positions,
)


@SPLITTERS.register("random")
def random_split(
    frame: pd.DataFrame, config: SplitConfig, seed: int, target: str | None
) -> SplitIndices:
    """Divide as linhas ao acaso (o alvo não é usado)."""
    _, n_validation, n_test = partition_sizes(len(frame), config)
    order = np.random.default_rng(seed).permutation(len(frame))
    return SplitIndices(
        train=sorted_positions(order[n_test + n_validation :]),
        validation=sorted_positions(order[n_test : n_test + n_validation]),
        test=sorted_positions(order[:n_test]),
    )
