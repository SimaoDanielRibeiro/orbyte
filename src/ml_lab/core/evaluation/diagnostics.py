"""Diagnósticos: o que a comparação entre treino e teste diz sobre o modelo.

Um modelo que acerta muito mais no treino do que no teste decorou os exemplos que
viu em vez de aprender a regra: é o overfitting. A diferença entre os dois relatórios
é o primeiro sinal. Os diagnósticos completos (desequilíbrio, curvas de
aprendizagem) chegam na fase 4.
"""

import math
from collections.abc import Mapping
from typing import Final

from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.report import EvaluationReport
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition
from ml_lab.formatting import format_number, format_percent

OVERFITTING_GAP: Final = 0.10
"""Diferença (em proporção) entre treino e teste a partir da qual se suspeita de overfitting."""

MIN_OVERFITTING_GAP: Final = 0.05
"""Abaixo desta diferença nunca é overfitting: um modelo conhece sempre melhor os seus exemplos."""

ERROR_RATIO: Final = 2.0
"""Quantas vezes o que falha no teste pode ser o que falha no treino antes de se suspeitar."""

NOISE_FACTOR: Final = 2.0
"""Quantos erros-padrão do teste a diferença tem de passar para não ser só a sorte da divisão."""

LUCKY_TEST_GAP: Final = 0.05
"""Diferença a favor do teste a partir da qual vale a pena desconfiar da divisão."""


def memorised(train: float, test: float, n_test: int | None = None) -> bool:
    """Se o modelo decorou o treino, numa métrica de 0 a 1 em que maior é melhor.

    A diferença sozinha engana quando quase tudo está certo: com classes desequilibradas, 100%
    no treino e 92,6% no teste parecem perto, mas o que falha passa de 0 para 7,4%, e 92,6% é o
    que se tem a responder sempre a classe maior. Por isso conta também o que falta para 1 (o
    erro, na exatidão; a variação por explicar, no R²): no teste, mais do dobro do do treino,
    com pelo menos 5 pontos de diferença (ADR 0051).

    Esta segunda regra só vale quando a diferença não cabe no ruído do teste: com as 30 linhas
    de teste do Iris, 3 erros dão 90% e 1 erro dá 96,7%, por isso 7,5 pontos de diferença ainda
    são sorte. O ruído mede-se como o de uma proporção, √(p(1 − p)/n), com p o que falha.

    Args:
        train: A métrica no treino.
        test: A mesma métrica nos dados que o modelo não viu (o teste, ou a validação).
        n_test: As linhas desses dados; sem elas (a média de uma validação cruzada, que usa o
            treino todo), não se desconta o ruído.
    """
    gap = train - test
    if gap >= OVERFITTING_GAP:
        return True
    missed = min(1.0, max(0.0, 1 - test))
    if gap < MIN_OVERFITTING_GAP or missed < ERROR_RATIO * (1 - train):
        return False
    if n_test is None or n_test <= 0:
        return True
    return gap >= NOISE_FACTOR * math.sqrt(missed * (1 - missed) / n_test)


def compare_train_and_test(reports: Mapping[Partition, EvaluationReport]) -> tuple[Notice, ...]:
    """Compara a métrica principal no treino e no teste."""
    train = reports.get(Partition.TRAIN)
    test = reports.get(Partition.TEST)
    if train is None or test is None:
        return ()
    name = PRIMARY_METRIC.get(train.task_type)
    if name is None:
        return ()
    train_result, test_result = train.find(name), test.find(name)
    if train_result is None or test_result is None:
        return ()
    if train_result.value is None or test_result.value is None:
        return ()
    if train_result.higher_is_better is None:  # só descreve (a proporção de anomalias)
        return ()
    gap = train_result.value - test_result.value
    if not train_result.higher_is_better:
        gap = -gap
    # O nome da métrica à cabeça ("Accuracy: …"): a frase serve a qualquer métrica principal.
    values = (
        f"{train_result.display_name}: {train_result.formatted} on the training data and "
        f"{test_result.formatted} on the test data"
    )
    points = format_number(abs(gap) * 100, 1)
    advice = "Try more regularisation or a simpler model."
    if gap >= OVERFITTING_GAP:
        return (
            Notice.warning(
                "overfitting",
                f"{values}, {points} points apart. The model seems to have memorised the "
                f"training data (overfitting). {advice}",
            ),
        )
    if train_result.higher_is_better and memorised(
        train_result.value, test_result.value, test.n_samples
    ):
        missed = _missed(train_result.value, as_percent=train_result.as_percent)
        on_test = _missed(test_result.value, as_percent=train_result.as_percent)
        return (
            Notice.warning(
                "overfitting",
                f"{values}: only {points} points apart, but what the model misses on the test "
                f"({on_test}) is more than twice what it misses on the training data "
                f"({missed}). It memorised part of the training data (overfitting). {advice}",
            ),
        )
    if gap <= -LUCKY_TEST_GAP:
        return (
            Notice.info(
                "lucky_test",
                f"{values}: the test came out better than the training. With few test rows, the "
                "luck of the split weighs a lot; try another seed.",
            ),
        )
    return (
        Notice.info(
            "no_overfitting",
            f"{values}: close values, no signs of overfitting.",
        ),
    )


def _missed(value: float, *, as_percent: bool) -> str:
    """O que falta para 1, escrito como a métrica (em percentagem, ou com três casas)."""
    missing = max(0.0, 1 - value)
    return format_percent(missing) if as_percent else format_number(missing, 3)
