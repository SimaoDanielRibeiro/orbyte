"""Explicação de cada métrica: o que mede e como se lê.

Os cartões de métricas das duas interfaces mostram estes textos, escritos uma só vez.
Um teste de contrato confirma que todas as métricas registadas têm explicação.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class MetricExplanation:
    """O que uma métrica mede e como se lê.

    Attributes:
        what: O que mede, numa frase.
        how_to_read: Como interpretar o valor, com os cuidados a ter.
    """

    what: str
    how_to_read: str


METRIC_EXPLANATIONS: Final[Mapping[str, MetricExplanation]] = MappingProxyType(
    {
        "accuracy": MetricExplanation(
            "A proporção de previsões certas.",
            "De 0% a 100%. É a mais intuitiva, mas engana com classes desequilibradas: se 95% "
            "dos exemplos são de uma classe, prever sempre essa classe dá 95% sem aprender nada.",
        ),
        "balanced_accuracy": MetricExplanation(
            "A média da recall de cada classe: cada classe pesa o mesmo.",
            "De 0% a 100%. Com duas classes, 50% é o que se consegue ao acaso. Se for muito "
            "menor do que a exatidão, o modelo está a falhar nas classes raras.",
        ),
        "precision": MetricExplanation(
            "Das vezes em que o modelo disse «é desta classe», quantas acertou.",
            "De 0% a 100%. Importa quando um falso alarme custa caro (por exemplo, marcar um "
            "email bom como spam). Em multiclasse, é a média das classes.",
        ),
        "recall": MetricExplanation(
            "Dos exemplos que eram desta classe, quantos o modelo encontrou.",
            "De 0% a 100%. Importa quando falhar um caso custa caro (por exemplo, um "
            "diagnóstico). Sobe facilmente à custa da precisão: dizer sempre «sim» dá 100%.",
        ),
        "f1": MetricExplanation(
            "A média harmónica da precisão e da recall.",
            "De 0% a 100%. Só é alta se as duas forem altas: resume o equilíbrio entre não dar "
            "falsos alarmes e não deixar casos escapar.",
        ),
        "log_loss": MetricExplanation(
            "Quão longe as probabilidades previstas estão da resposta certa.",
            "Menor é melhor; 0 seria a certeza absoluta e sempre certa. Castiga muito as "
            "previsões erradas feitas com confiança. Só existe para modelos que dão "
            "probabilidades.",
        ),
        "roc_auc": MetricExplanation(
            "A probabilidade de o modelo dar mais pontuação a um exemplo da classe positiva do "
            "que a um da negativa.",
            "De 0,5 (ao acaso) a 1 (separação perfeita). Não depende do limiar de decisão, por "
            "isso compara bem modelos. Em multiclasse, é a média de cada classe contra as "
            "outras.",
        ),
    }
)
