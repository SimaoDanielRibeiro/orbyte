"""Explicações didáticas da avaliação, da explicabilidade e do Laboratório (painel "Porquê?").

Como as das outras camadas, vivem no backend para as duas interfaces mostrarem o mesmo
texto, escrito uma só vez; ``services/explanations.py`` junta-as todas.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

EVALUATION_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "cross_validation": Explanation(
            "Validação cruzada",
            "Um só teste pode sair melhor ou pior por sorte (as linhas que calharam no teste). "
            "A validação cruzada parte o treino em k folds e treina k vezes, cada uma avaliada "
            "num fold diferente: a média estima o desempenho em dados novos e o desvio padrão "
            "diz quanto ele varia. As transformações aprendem de novo em cada fold, para o fold "
            "de fora nunca influenciar o treino.",
        ),
        "learning_curve": Explanation(
            "Curva de aprendizagem",
            "Treina-se com cada vez mais linhas e compara-se o treino com a avaliação. Curvas "
            "afastadas no fim: o modelo decora (variância), e mais dados ou mais regularização "
            "ajudam. Curvas juntas mas fracas: o modelo é simples demais (viés). A avaliação "
            "ainda a subir no fim: mais dados ainda ajudariam.",
        ),
        "roc_curve": Explanation(
            "Curvas ROC e de precisão-recall",
            "Um classificador com probabilidades decide com um limiar (por omissão, 50%). Cada "
            "limiar dá uma combinação de acertos e falsos alarmes; a curva ROC mostra-as todas, "
            "e a área por baixo dela (AUC) resume-as num número que não depende do limiar. Com "
            "classes muito desequilibradas, a curva de precisão-recall é mais honesta.",
        ),
        "residuals": Explanation(
            "Resíduos",
            "O resíduo é o erro de cada previsão: real menos previsto. Num bom modelo, os "
            "resíduos parecem ruído à volta de zero. Uma curva nos resíduos quer dizer que o "
            "modelo não apanhou uma relação curva; um leque a abrir quer dizer que o erro "
            "cresce com o valor (talvez valha a pena prever o logaritmo do alvo).",
        ),
        "silhouette": Explanation(
            "Silhueta",
            "Para cada ponto, compara a distância média aos pontos do seu grupo com a distância "
            "média aos do grupo vizinho mais próximo: perto de 1, o ponto está bem no seu "
            "grupo; perto de 0, está na fronteira; negativa, estaria melhor noutro. O perfil "
            "mostra todos os pontos de cada grupo, por ordem: grupos com muitas barras curtas "
            "ou negativas estão mal definidos.",
        ),
        "reference": Explanation(
            "Coluna de referência",
            "Nas tarefas sem alvo, uma coluna-alvo escolhida no ecrã Preparar não entra no "
            "treino: fica como referência, para no fim comparar o que o modelo encontrou com "
            "ela (os grupos do k-means com as espécies, as anomalias marcadas com as "
            "verdadeiras). É uma forma honesta de avaliar aprendizagem não supervisionada.",
        ),
        "permutation_importance": Explanation(
            "Importância por permutação",
            "Para saber quanto o modelo depende de uma feature, baralham-se os seus valores no "
            "teste e mede-se quanto a métrica piora. Funciona com qualquer modelo e mede no "
            "teste, por isso diz o que o modelo usa para generalizar. Com features "
            "correlacionadas, baralhar uma custa pouco: a outra substitui-a.",
        ),
        "partial_dependence": Explanation(
            "Dependência parcial",
            "Mostra como a previsão muda, em média, quando uma feature muda e as outras ficam "
            "como estão. As linhas finas (ICE) fazem o mesmo para linhas individuais: se forem "
            "paralelas, o efeito é igual para todas; se se cruzarem, a feature interage com "
            "outras. Cuidado com combinações que não existem nos dados.",
        ),
        "local_explanation": Explanation(
            "Explicar uma previsão",
            "Nos modelos lineares, a previsão é a ordenada mais a soma de peso × valor de cada "
            "feature: cada parcela é o contributo exato da feature. Nos outros modelos, mede-se "
            "quanto a previsão muda quando cada feature passa a valer a média do treino: uma "
            "aproximação, que ignora as interações entre features.",
        ),
        "bias_variance": Explanation(
            "Viés e variância",
            "Um modelo simples demais erra sempre da mesma maneira (viés): fica mal no treino e "
            "no teste. Um modelo flexível demais segue o ruído do treino (variância): fica ótimo "
            "no treino e mal no teste. O melhor está no meio, e encontra-se olhando para o "
            "teste, nunca para o treino.",
        ),
        "learning_rate_demo": Explanation(
            "Taxa de aprendizagem",
            "O erro da regressão linear é uma taça com o fundo nos melhores pesos. O gradiente "
            "descendente desce pela parte mais inclinada, com passos do tamanho da taxa. "
            "Pequena demais, demora uma eternidade; grande demais, salta de um lado para o "
            "outro da taça; maior ainda, salta para cada vez mais alto e explode.",
        ),
        "knn_k": Explanation(
            "O k do KNN",
            "Com k = 1, cada ponto de treino decide sozinho a sua vizinhança: acerta em 100% "
            "do treino e desenha ilhas à volta do ruído. Com k grande, a votação junta pontos "
            "longe demais e a fronteira fica lisa demais. O k certo equilibra os dois, e "
            "escolhe-se pelo teste (ou com validação cruzada).",
        ),
        "regularization": Explanation(
            "Regularização",
            "Somar ao erro um custo pelo tamanho dos pesos obriga o modelo a ser mais simples. "
            "Na Ridge (L2) todos os pesos encolhem; no Lasso (L1) os menos úteis chegam "
            "exatamente a zero. Alpha a mais apaga também o que importa; o melhor alpha "
            "escolhe-se pelo teste.",
        ),
    }
)
