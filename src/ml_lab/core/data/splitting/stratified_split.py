"""Divisão estratificada: mantém a proporção de cada classe em todas as partes.

Numa divisão aleatória, uma classe rara pode calhar quase toda no treino e quase
nenhuma no teste, e a avaliação deixa de dizer a verdade sobre ela. A
estratificada divide cada classe em separado, com as mesmas proporções, por isso
treino, validação e teste ficam com a mesma mistura de classes.

As quantidades por classe são arredondadas pelo método do maior resto: a soma bate
certo com o total pedido, e nenhuma classe fica sem linhas no treino.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_lab.core.data.splitting.base import (
    SPLITTERS,
    SplitConfig,
    SplitIndices,
    partition_sizes,
    sorted_positions,
)
from ml_lab.core.foundation.errors import DataError, InvalidParameterError


@SPLITTERS.register("stratified")
def stratified_split(
    frame: pd.DataFrame, config: SplitConfig, seed: int, target: str | None
) -> SplitIndices:
    """Divide cada classe do alvo nas mesmas proporções.

    Raises:
        InvalidParameterError: Se não houver alvo.
        DataError: Se o alvo tiver nulos ou alguma classe tiver uma só linha.
    """
    if target is None:
        raise InvalidParameterError(
            "A divisão estratificada precisa de um alvo: escolhe a coluna-alvo ou usa a "
            "divisão aleatória."
        )
    labels = frame[target]
    if labels.isna().any():
        raise DataError(
            f"O alvo {target} tem valores em falta: remove essas linhas antes de estratificar."
        )
    codes, classes = pd.factorize(labels, sort=True)
    counts = np.bincount(codes, minlength=len(classes))
    tiny = [str(classes[int(c)]) for c in np.flatnonzero(counts < 2)]
    if tiny:
        raise DataError(
            f"As classes {', '.join(tiny)} só têm uma linha, por isso não se podem repartir. "
            "Remove-as ou usa a divisão aleatória."
        )

    _, n_validation, n_test = partition_sizes(len(frame), config)
    test_counts = allocate(counts, n_test, capacity=counts - 1)
    validation_counts = allocate(counts, n_validation, capacity=counts - 1 - test_counts)

    rng = np.random.default_rng(seed)
    parts: dict[str, list[NDArray[np.intp]]] = {"train": [], "validation": [], "test": []}
    for code, (n_class_test, n_class_validation) in enumerate(
        zip(test_counts, validation_counts, strict=True)
    ):
        rows = rng.permutation(np.flatnonzero(codes == code))
        parts["test"].append(rows[:n_class_test])
        parts["validation"].append(rows[n_class_test : n_class_test + n_class_validation])
        parts["train"].append(rows[n_class_test + n_class_validation :])
    return SplitIndices(
        **{name: sorted_positions(np.concatenate(pieces)) for name, pieces in parts.items()}
    )


def allocate(
    counts: NDArray[np.int64], total: int, *, capacity: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Reparte ``total`` linhas pelas classes, em proporção do tamanho de cada uma.

    Método do maior resto: cada classe recebe a parte inteira da sua quota, e as
    linhas que sobram vão para as classes com as maiores partes decimais. Nenhuma
    classe recebe mais do que a sua ``capacity``.

    >>> allocate(np.array([50, 30, 20]), 10, capacity=np.array([49, 29, 19])).tolist()
    [5, 3, 2]
    """
    quotas = counts * total / counts.sum()
    allocation = np.minimum(np.floor(quotas).astype(np.int64), capacity)
    remainders = quotas - np.floor(quotas)
    # Ordem estável: em caso de empate, a primeira classe recebe primeiro.
    for code in np.argsort(-remainders, kind="stable"):
        if allocation.sum() >= total:
            break
        if allocation[code] < capacity[code]:
            allocation[code] += 1
    return allocation
