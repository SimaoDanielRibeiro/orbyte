"""Diagnósticos: o que a comparação entre treino e teste diz sobre o modelo.

Um modelo que acerta muito mais no treino do que no teste decorou os exemplos que
viu em vez de aprender a regra: é o overfitting. A diferença entre os dois relatórios
é o primeiro sinal. Os diagnósticos completos (desequilíbrio, curvas de
aprendizagem) chegam na fase 4.
"""

from collections.abc import Mapping
from typing import Final

from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.report import EvaluationReport
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition
from ml_lab.formatting import format_number

OVERFITTING_GAP: Final = 0.10
"""Diferença (em proporção) entre treino e teste a partir da qual se suspeita de overfitting."""

LUCKY_TEST_GAP: Final = 0.05
"""Diferença a favor do teste a partir da qual vale a pena desconfiar da divisão."""


def compare_train_and_test(reports: Mapping[Partition, EvaluationReport]) -> tuple[Notice, ...]:
    """Compara a métrica principal no treino e no teste."""
    train = reports.get(Partition.TRAIN)
    test = reports.get(Partition.TEST)
    if train is None or test is None:
        return ()
    name = PRIMARY_METRIC.get(train.task_type)
    if name is None:
        return ()
    train_result, test_result = train.metric(name), test.metric(name)
    if train_result.value is None or test_result.value is None:
        return ()
    gap = train_result.value - test_result.value
    if not train_result.higher_is_better:
        gap = -gap
    label = train_result.display_name.lower()
    values = f"{train_result.formatted} no treino e {test_result.formatted} no teste"
    points = format_number(abs(gap) * 100, 1)
    if gap >= OVERFITTING_GAP:
        return (
            Notice.warning(
                "overfitting",
                f"A {label} é {values}: {points} pontos de diferença. O modelo parece ter "
                "decorado o treino (overfitting). Experimenta mais regularização ou um modelo "
                "mais simples.",
            ),
        )
    if gap <= -LUCKY_TEST_GAP:
        return (
            Notice.info(
                "lucky_test",
                f"A {label} é {values}: o teste saiu melhor do que o treino. Com poucas linhas de "
                "teste, a sorte da divisão pesa; experimenta outra seed.",
            ),
        )
    return (
        Notice.info(
            "no_overfitting",
            f"A {label} é {values}: valores próximos, sem sinais de overfitting.",
        ),
    )
