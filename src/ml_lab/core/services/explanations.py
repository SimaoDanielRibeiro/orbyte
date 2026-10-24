"""O painel "Porquê?": as explicações de todos os temas, venham de que camada vierem.

Cada camada escreve as suas (os dados em ``core/data/explanations.py``, o treino em
``core/training/explanations.py``); a interface pede-as aqui pelo tema.
"""

from collections import ChainMap
from collections.abc import Mapping
from typing import Final

from ml_lab.core.data.explanations import EXPLANATIONS as DATA_EXPLANATIONS
from ml_lab.core.data.explanations import Explanation
from ml_lab.core.foundation.errors import ComponentNotFoundError
from ml_lab.core.training.explanations import TRAINING_EXPLANATIONS

EXPLANATIONS: Final[Mapping[str, Explanation]] = ChainMap(
    dict(DATA_EXPLANATIONS), dict(TRAINING_EXPLANATIONS)
)


def explain(topic: str) -> Explanation:
    """A explicação de um tema (``"split"``, ``"loss"``...).

    Raises:
        ComponentNotFoundError: Se o tema não existir.
    """
    try:
        return EXPLANATIONS[topic]
    except KeyError:
        raise ComponentNotFoundError("explanations", topic, EXPLANATIONS) from None
