"""Explicação de cada métrica: o que mede, como se lê e a fórmula.

Os cartões de métricas das duas interfaces mostram estes textos, escritos uma só vez.
Um teste de contrato confirma que todas as métricas registadas têm explicação, com a
fórmula e o significado de cada símbolo.

As fórmulas estão em LaTeX (a interface aprimorada mostra-as com o KaTeX). Os símbolos que
uma métrica dá como termos, com os números de um run (``Metric.breakdown``), aparecem
escritos na fórmula exatamente da mesma maneira.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class FormulaSymbol:
    """Um símbolo da fórmula e o que quer dizer.

    Attributes:
        symbol: O símbolo, em LaTeX.
        meaning: O que é, em português (pode ter fórmulas curtas entre ``$``).
    """

    symbol: str
    meaning: str


@dataclass(frozen=True, slots=True)
class MetricExplanation:
    """O que uma métrica mede, como se lê e como se calcula.

    Attributes:
        what: O que mede, numa frase.
        how_to_read: Como interpretar o valor, com os cuidados a ter.
        formula: A definição, em LaTeX.
        symbols: O que quer dizer cada símbolo da fórmula.
    """

    what: str
    how_to_read: str
    formula: str
    symbols: tuple[FormulaSymbol, ...]


_N = FormulaSymbol("n", "o número de exemplos avaliados")
_Y = FormulaSymbol("y_i", "o valor real do exemplo $i$")
_Y_HAT = FormulaSymbol(r"\hat{y}_i", "a previsão do modelo para o exemplo $i$")
_TRUE_POSITIVES = FormulaSymbol(
    r"\mathrm{VP}",
    "verdadeiros positivos: os exemplos da classe que o modelo disse serem dela",
)
_FALSE_POSITIVES = FormulaSymbol(
    r"\mathrm{FP}",
    "falsos positivos: os exemplos de outras classes que o modelo disse serem desta",
)
_FALSE_NEGATIVES = FormulaSymbol(
    r"\mathrm{FN}",
    "falsos negativos: os exemplos da classe que o modelo deu como de outra",
)


METRIC_EXPLANATIONS: Final[Mapping[str, MetricExplanation]] = MappingProxyType(
    {
        "accuracy": MetricExplanation(
            "A proporção de previsões certas.",
            "De 0% a 100%. É a mais intuitiva, mas engana com classes desequilibradas: se 95% "
            "dos exemplos são de uma classe, prever sempre essa classe dá 95% sem aprender nada.",
            formula=r"\text{exatidão} = \frac{\text{acertos}}{n}",
            symbols=(
                FormulaSymbol(r"\text{acertos}", "o número de previsões certas"),
                _N,
            ),
        ),
        "balanced_accuracy": MetricExplanation(
            "A média da recall de cada classe: cada classe pesa o mesmo.",
            "De 0% a 100%. Com duas classes, 50% é o que se consegue ao acaso. Se for muito "
            "menor do que a exatidão, o modelo está a falhar nas classes raras.",
            formula=r"\text{exatidão equilibrada} = \frac{1}{K} \sum_{c=1}^{K} \text{recall}_c",
            symbols=(
                FormulaSymbol("K", "o número de classes que aparecem nos dados avaliados"),
                FormulaSymbol(
                    r"\text{recall}_c",
                    "a recall da classe $c$: a parte dos exemplos dela que o modelo encontrou",
                ),
            ),
        ),
        "precision": MetricExplanation(
            "Das vezes em que o modelo disse «é desta classe», quantas acertou.",
            "De 0% a 100%. Importa quando um falso alarme custa caro (por exemplo, marcar um "
            "email bom como spam). Em multiclasse, é a média das classes.",
            formula=r"\text{precisão} = \frac{\mathrm{VP}}{\mathrm{VP} + \mathrm{FP}}",
            symbols=(_TRUE_POSITIVES, _FALSE_POSITIVES),
        ),
        "recall": MetricExplanation(
            "Dos exemplos que eram desta classe, quantos o modelo encontrou.",
            "De 0% a 100%. Importa quando falhar um caso custa caro (por exemplo, um "
            "diagnóstico). Sobe facilmente à custa da precisão: dizer sempre «sim» dá 100%.",
            formula=r"\text{recall} = \frac{\mathrm{VP}}{\mathrm{VP} + \mathrm{FN}}",
            symbols=(_TRUE_POSITIVES, _FALSE_NEGATIVES),
        ),
        "f1": MetricExplanation(
            "A média harmónica da precisão e da recall.",
            "De 0% a 100%. Só é alta se as duas forem altas: resume o equilíbrio entre não dar "
            "falsos alarmes e não deixar casos escapar.",
            formula=r"F_1 = 2 \cdot \frac{P \cdot R}{P + R}",
            symbols=(
                FormulaSymbol("P", "a precisão"),
                FormulaSymbol("R", "a recall"),
            ),
        ),
        "log_loss": MetricExplanation(
            "Quão longe as probabilidades previstas estão da resposta certa.",
            "Menor é melhor; 0 seria a certeza absoluta e sempre certa. Castiga muito as "
            "previsões erradas feitas com confiança. Só existe para modelos que dão "
            "probabilidades.",
            formula=r"\text{log-loss} = \frac{1}{n} \sum_{i=1}^{n} -\log p_i",
            symbols=(
                FormulaSymbol(
                    "p_i", "a probabilidade que o modelo deu à classe certa do exemplo $i$"
                ),
                _N,
            ),
        ),
        "r2": MetricExplanation(
            "A proporção da variação do alvo que o modelo explica.",
            "1 é perfeito; 0 é o mesmo que prever sempre a média; abaixo de 0, o modelo é pior "
            "do que isso. Não tem unidades, por isso compara modelos em alvos diferentes.",
            formula=r"R^2 = 1 - \frac{\mathrm{SQ}_{\text{res}}}{\mathrm{SQ}_{\text{tot}}}",
            symbols=(
                FormulaSymbol(
                    r"\mathrm{SQ}_{\text{res}}",
                    "a soma dos quadrados dos resíduos: $\\sum_i (y_i - \\hat{y}_i)^2$",
                ),
                FormulaSymbol(
                    r"\mathrm{SQ}_{\text{tot}}",
                    "a soma dos quadrados à volta da média: $\\sum_i (y_i - \\bar{y})^2$, o erro "
                    "de prever sempre a média",
                ),
            ),
        ),
        "rmse": MetricExplanation(
            "A raiz do erro quadrático médio (RMSE, do inglês *root mean squared error*): um "
            "erro típico das previsões, nas unidades do alvo.",
            "Menor é melhor. Como os erros entram ao quadrado, poucos erros grandes pesam muito: "
            "compara-a com o erro absoluto médio para ver se há previsões muito falhadas.",
            formula=r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}",
            symbols=(_Y, _Y_HAT, _N),
        ),
        "mae": MetricExplanation(
            "O erro absoluto médio (MAE, do inglês *mean absolute error*): a distância média "
            "entre a previsão e o valor real, nas unidades do alvo.",
            "Menor é melhor. Lê-se diretamente: «em média, o modelo falha por tanto». Todos os "
            "erros pesam o mesmo, grandes ou pequenos.",
            formula=r"\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|",
            symbols=(_Y, _Y_HAT, _N),
        ),
        "mse": MetricExplanation(
            "O erro quadrático médio (MSE, do inglês *mean squared error*): a média dos erros "
            "ao quadrado, a loss que a regressão linear minimiza.",
            "Menor é melhor, mas as unidades são as do alvo ao quadrado, o que a torna difícil "
            "de ler; para interpretar, usa a raiz (RMSE).",
            formula=r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2",
            symbols=(_Y, _Y_HAT, _N),
        ),
        "silhouette": MetricExplanation(
            "Para cada ponto, quão mais perto está dos pontos do seu grupo do que dos do grupo "
            "vizinho, em média.",
            "De -1 a 1. Perto de 1: grupos compactos e bem separados; perto de 0: grupos "
            "encostados uns aos outros; negativa: há pontos no grupo errado. Serve para "
            "comparar valores de k diferentes.",
            formula=r"s = \frac{1}{n} \sum_{i=1}^{n} \frac{b_i - a_i}{\max(a_i, b_i)}",
            symbols=(
                FormulaSymbol("a_i", "a distância média do ponto $i$ aos pontos do seu grupo"),
                FormulaSymbol(
                    "b_i",
                    "a distância média do ponto $i$ aos pontos do grupo vizinho mais próximo",
                ),
                FormulaSymbol("n", "o número de pontos avaliados (sem o ruído)"),
            ),
        ),
        "davies_bouldin": MetricExplanation(
            "Quão parecido é cada grupo com o seu vizinho mais próximo: a dispersão dentro dos "
            "grupos sobre a distância entre eles.",
            "Menor é melhor; 0 seria grupos infinitamente compactos e afastados. Como a "
            "silhueta, prefere grupos redondos.",
            formula=(
                r"\mathrm{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} "
                r"\frac{s_i + s_j}{d_{ij}}"
            ),
            symbols=(
                FormulaSymbol("k", "o número de grupos"),
                FormulaSymbol("s_i", "a distância média dos pontos do grupo $i$ ao seu centro"),
                FormulaSymbol("d_{ij}", "a distância entre os centros dos grupos $i$ e $j$"),
            ),
        ),
        "calinski_harabasz": MetricExplanation(
            "A variância entre os grupos sobre a variância dentro dos grupos.",
            "Maior é melhor, mas não tem escala fixa: só compara agrupamentos dos mesmos dados "
            "(por exemplo, k = 3 contra k = 4).",
            formula=r"\mathrm{CH} = \frac{B / (k - 1)}{W / (n - k)}",
            symbols=(
                FormulaSymbol(
                    "B",
                    "a dispersão entre os grupos: para cada grupo, o número de pontos vezes a "
                    "distância ao quadrado do seu centro ao centro de todos os pontos",
                ),
                FormulaSymbol(
                    "W",
                    "a dispersão dentro dos grupos: a soma das distâncias ao quadrado de cada "
                    "ponto ao centro do seu grupo",
                ),
                FormulaSymbol("k", "o número de grupos"),
                FormulaSymbol("n", "o número de pontos"),
            ),
        ),
        "n_clusters": MetricExplanation(
            "Quantos grupos ficaram com pontos (o ruído não conta).",
            "Não é bom nem mau: no k-means é o k que escolheste; no DBSCAN, é o que a "
            "densidade dos dados ditou.",
            formula=r"k = \left| \{ g_i : g_i \neq \text{ruído} \} \right|",
            symbols=(FormulaSymbol("g_i", "o grupo em que ficou o ponto $i$"),),
        ),
        "noise_ratio": MetricExplanation(
            "A proporção de pontos que não ficaram em nenhum grupo (só o DBSCAN tem ruído).",
            "Muito ruído (mais de metade) quer dizer um raio pequeno demais; nenhum ruído pode "
            "querer dizer um raio grande demais, com tudo num só grupo.",
            formula=r"\text{ruído} = \frac{n_{\text{ruído}}}{n}",
            symbols=(
                FormulaSymbol(r"n_{\text{ruído}}", "os pontos que não ficaram em nenhum grupo"),
                FormulaSymbol("n", "o número de pontos avaliados"),
            ),
        ),
        "adjusted_rand": MetricExplanation(
            "O acordo entre os grupos encontrados e a coluna de referência, descontado o que se "
            "esperava ao acaso.",
            "1 é um acordo perfeito; 0 é o mesmo que agrupar ao acaso. Não importa o número que "
            "cada grupo recebe, só quem fica com quem.",
            formula=(
                r"\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}"
                r"{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]}"
            ),
            symbols=(
                FormulaSymbol(
                    r"\mathrm{RI}",
                    "o índice de Rand: a parte dos pares de pontos em que os grupos e a "
                    "referência concordam (juntos nos dois, ou separados nos dois)",
                ),
                FormulaSymbol(
                    r"\mathbb{E}[\mathrm{RI}]", "o valor que o índice teria com grupos ao acaso"
                ),
            ),
        ),
        "trustworthiness": MetricExplanation(
            "Se os vizinhos de cada ponto na projeção eram mesmo vizinhos nos dados originais "
            "(os 5 mais próximos).",
            "De 0% a 100%. Perto de 100%: o que está junto no gráfico está junto nos dados. Não "
            "garante o contrário: pontos afastados no gráfico podem ser vizinhos nos dados.",
            formula=(
                r"T = 1 - \frac{2}{n k (2n - 3k - 1)} \sum_{i=1}^{n} \sum_{j \in U_i} "
                r"(r_{ij} - k)"
            ),
            symbols=(
                FormulaSymbol("k", "o número de vizinhos que se comparam (5)"),
                FormulaSymbol(
                    "U_i",
                    "os pontos que são vizinhos de $i$ na projeção, mas não o eram nos dados "
                    "originais",
                ),
                FormulaSymbol(
                    "r_{ij}", "a posição de $j$ entre os vizinhos de $i$ nos dados originais"
                ),
                FormulaSymbol("n", "o número de pontos"),
            ),
        ),
        "anomaly_rate": MetricExplanation(
            "A proporção de pontos marcados como anomalias.",
            "Não é boa nem má: anda perto da proporção esperada que escolheste nos "
            "hiperparâmetros (no treino, por construção).",
            formula=r"\text{marcadas} = \frac{n_{\text{marcados}}}{n}",
            symbols=(
                FormulaSymbol(
                    r"n_{\text{marcados}}", "os pontos que o detetor marcou como anomalias"
                ),
                FormulaSymbol("n", "o número de pontos avaliados"),
            ),
        ),
        "anomaly_precision": MetricExplanation(
            "Dos pontos marcados como anomalias, quantos o eram de facto (segundo a coluna de "
            "referência).",
            "De 0% a 100%. Baixa quer dizer muitos falsos alarmes: o limiar está demasiado "
            "baixo, ou as anomalias não se distinguem nas features.",
            formula=r"\text{precisão} = \frac{\mathrm{VP}}{\mathrm{VP} + \mathrm{FP}}",
            symbols=(
                FormulaSymbol(r"\mathrm{VP}", "as anomalias verdadeiras que foram marcadas"),
                FormulaSymbol(r"\mathrm{FP}", "os pontos normais marcados como anomalias"),
            ),
        ),
        "anomaly_recall": MetricExplanation(
            "Das anomalias verdadeiras, quantas foram marcadas.",
            "De 0% a 100%. Sobe se esperares mais anomalias (o limiar desce), à custa da precisão.",
            formula=r"\text{recall} = \frac{\mathrm{VP}}{\mathrm{VP} + \mathrm{FN}}",
            symbols=(
                FormulaSymbol(r"\mathrm{VP}", "as anomalias verdadeiras que foram marcadas"),
                FormulaSymbol(r"\mathrm{FN}", "as anomalias verdadeiras que ficaram por marcar"),
            ),
        ),
        "anomaly_roc_auc": MetricExplanation(
            "A probabilidade de uma anomalia verdadeira ter uma pontuação mais alta do que um "
            "ponto normal.",
            "De 0,5 (ao acaso) a 1 (separação perfeita). Não depende do limiar: diz se a "
            "pontuação ordena bem os pontos, mesmo que o corte esteja no sítio errado.",
            formula=r"\text{AUC} = P\left(s_{\text{anomalia}} > s_{\text{normal}}\right)",
            symbols=(
                FormulaSymbol(
                    r"s_{\text{anomalia}}", "a pontuação de uma anomalia verdadeira, ao acaso"
                ),
                FormulaSymbol(r"s_{\text{normal}}", "a pontuação de um ponto normal, ao acaso"),
            ),
        ),
        "roc_auc": MetricExplanation(
            "A probabilidade de o modelo dar mais pontuação a um exemplo da classe positiva do "
            "que a um da negativa.",
            "De 0,5 (ao acaso) a 1 (separação perfeita). Não depende do limiar de decisão, por "
            "isso compara bem modelos. Em multiclasse, é a média de cada classe contra as "
            "outras.",
            formula=r"\text{AUC} = \int_0^1 \mathrm{TVP} \; d\,\mathrm{TFP}",
            symbols=(
                FormulaSymbol(
                    r"\mathrm{TVP}",
                    "a taxa de verdadeiros positivos (a recall) com cada limiar de decisão",
                ),
                FormulaSymbol(
                    r"\mathrm{TFP}",
                    "a taxa de falsos positivos: a parte dos negativos dados como positivos",
                ),
            ),
        ),
    }
)
