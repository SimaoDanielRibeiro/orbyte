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
        "r2": MetricExplanation(
            "A proporção da variação do alvo que o modelo explica.",
            "1 é perfeito; 0 é o mesmo que prever sempre a média; abaixo de 0, o modelo é pior "
            "do que isso. Não tem unidades, por isso compara modelos em alvos diferentes.",
        ),
        "rmse": MetricExplanation(
            "A raiz do erro quadrático médio (RMSE, do inglês *root mean squared error*): um "
            "erro típico das previsões, nas unidades do alvo.",
            "Menor é melhor. Como os erros entram ao quadrado, poucos erros grandes pesam muito: "
            "compara-a com o erro absoluto médio para ver se há previsões muito falhadas.",
        ),
        "mae": MetricExplanation(
            "O erro absoluto médio (MAE, do inglês *mean absolute error*): a distância média "
            "entre a previsão e o valor real, nas unidades do alvo.",
            "Menor é melhor. Lê-se diretamente: «em média, o modelo falha por tanto». Todos os "
            "erros pesam o mesmo, grandes ou pequenos.",
        ),
        "mse": MetricExplanation(
            "O erro quadrático médio (MSE, do inglês *mean squared error*): a média dos erros "
            "ao quadrado, a loss que a regressão linear minimiza.",
            "Menor é melhor, mas as unidades são as do alvo ao quadrado, o que a torna difícil "
            "de ler; para interpretar, usa a raiz (RMSE).",
        ),
        "silhouette": MetricExplanation(
            "Para cada ponto, quão mais perto está dos pontos do seu grupo do que dos do grupo "
            "vizinho, em média.",
            "De -1 a 1. Perto de 1: grupos compactos e bem separados; perto de 0: grupos "
            "encostados uns aos outros; negativa: há pontos no grupo errado. Serve para "
            "comparar valores de k diferentes.",
        ),
        "davies_bouldin": MetricExplanation(
            "Quão parecido é cada grupo com o seu vizinho mais próximo: a dispersão dentro dos "
            "grupos sobre a distância entre eles.",
            "Menor é melhor; 0 seria grupos infinitamente compactos e afastados. Como a "
            "silhueta, prefere grupos redondos.",
        ),
        "calinski_harabasz": MetricExplanation(
            "A variância entre os grupos sobre a variância dentro dos grupos.",
            "Maior é melhor, mas não tem escala fixa: só compara agrupamentos dos mesmos dados "
            "(por exemplo, k = 3 contra k = 4).",
        ),
        "n_clusters": MetricExplanation(
            "Quantos grupos ficaram com pontos (o ruído não conta).",
            "Não é bom nem mau: no k-means é o k que escolheste; no DBSCAN, é o que a "
            "densidade dos dados ditou.",
        ),
        "noise_ratio": MetricExplanation(
            "A proporção de pontos que não ficaram em nenhum grupo (só o DBSCAN tem ruído).",
            "Muito ruído (mais de metade) quer dizer um raio pequeno demais; nenhum ruído pode "
            "querer dizer um raio grande demais, com tudo num só grupo.",
        ),
        "adjusted_rand": MetricExplanation(
            "O acordo entre os grupos encontrados e a coluna de referência, descontado o que se "
            "esperava ao acaso.",
            "1 é um acordo perfeito; 0 é o mesmo que agrupar ao acaso. Não importa o número que "
            "cada grupo recebe, só quem fica com quem.",
        ),
        "trustworthiness": MetricExplanation(
            "Se os vizinhos de cada ponto na projeção eram mesmo vizinhos nos dados originais "
            "(os 5 mais próximos).",
            "De 0% a 100%. Perto de 100%: o que está junto no gráfico está junto nos dados. Não "
            "garante o contrário: pontos afastados no gráfico podem ser vizinhos nos dados.",
        ),
        "anomaly_rate": MetricExplanation(
            "A proporção de pontos marcados como anomalias.",
            "Não é boa nem má: anda perto da proporção esperada que escolheste nos "
            "hiperparâmetros (no treino, por construção).",
        ),
        "anomaly_precision": MetricExplanation(
            "Dos pontos marcados como anomalias, quantos o eram de facto (segundo a coluna de "
            "referência).",
            "De 0% a 100%. Baixa quer dizer muitos falsos alarmes: o limiar está demasiado "
            "baixo, ou as anomalias não se distinguem nas features.",
        ),
        "anomaly_recall": MetricExplanation(
            "Das anomalias verdadeiras, quantas foram marcadas.",
            "De 0% a 100%. Sobe se esperares mais anomalias (o limiar desce), à custa da precisão.",
        ),
        "anomaly_roc_auc": MetricExplanation(
            "A probabilidade de uma anomalia verdadeira ter uma pontuação mais alta do que um "
            "ponto normal.",
            "De 0,5 (ao acaso) a 1 (separação perfeita). Não depende do limiar: diz se a "
            "pontuação ordena bem os pontos, mesmo que o corte esteja no sítio errado.",
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
