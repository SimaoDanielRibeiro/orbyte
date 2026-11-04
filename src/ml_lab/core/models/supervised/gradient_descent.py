"""O que os modelos didáticos treinados por gradiente descendente partilham.

Com uma taxa de aprendizagem grande demais (ou features em escalas muito diferentes,
sem padronização), cada passo salta por cima do mínimo e cai mais longe do que estava:
a loss cresce de época para época, multiplicada sempre pelo mesmo fator. Se o treino
continuasse, os números acabariam por transbordar (``inf``) e as previsões deixavam de
ser números. Por isso o treino pára assim que a loss fica muito pior do que no início.
"""

import math
from typing import Final

from ml_lab.core.foundation.notices import Notice
from ml_lab.formatting import format_number

DIVERGENCE_FACTOR: Final = 1e6
"""A loss é dada como explodida quando fica este número de vezes maior do que no início."""


def has_diverged(loss: float, initial_loss: float) -> bool:
    """Se a loss explodiu: deixou de ser um número, ou ficou muito pior do que no início."""
    return not math.isfinite(loss) or loss > DIVERGENCE_FACTOR * max(initial_loss, 1e-12)


def diverged_notice(learning_rate: float) -> Notice:
    """O aviso a dar quando o treino pára porque a loss explodiu."""
    return Notice.warning(
        "diverged",
        f"The loss blew up with a learning rate of {format_number(learning_rate, 4)}: each step "
        "jumped over the minimum and landed further away. Try a smaller rate (and standardise "
        "the features in the preparation).",
    )
