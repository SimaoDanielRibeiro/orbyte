"""Explicações didáticas da avaliação, da explicabilidade e do Laboratório (painel "Porquê?").

Como as das outras camadas, vivem no backend, escritas uma só vez, e a interface pede-as pela
API; ``services/explanations.py`` junta-as todas.
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
        # Como ler os gráficos da avaliação: a interface mostra-os por baixo deles.
        "train_vs_test": Explanation(
            "Treino e teste",
            "O modelo aprendeu com o treino; o teste mostra como se sai com dados que nunca "
            "viu. Uma diferença grande a favor do treino é o sinal de overfitting.",
        ),
        "confusion_matrix": Explanation(
            "Matriz de confusão",
            "Cada linha é a classe real e cada coluna a prevista: a diagonal são os acertos, o "
            "resto são as confusões do modelo.",
        ),
        "reference_matrix": Explanation(
            "Grupos e referência",
            "Cada linha é uma classe da coluna de referência e cada coluna um grupo "
            "encontrado: grupos que correspondem a classes têm quase tudo numa só célula da "
            "coluna.",
        ),
        "anomaly_matrix": Explanation(
            "Anomalias e referência",
            "Cada linha é o que o ponto era (segundo a coluna de referência) e cada coluna o "
            "que o detetor disse: a diagonal são os acertos.",
        ),
        "roc_pr_chart": Explanation(
            "Como ler as curvas",
            "Cada ponto de uma curva é um limiar de decisão diferente. A ROC compara os "
            "acertos com os falsos alarmes (a diagonal é o acaso); a de precisão-recall mostra "
            "o preço de encontrar mais casos. Em multiclasse, há uma curva por classe contra "
            "as outras.",
        ),
        "errors_map": Explanation(
            "Onde o modelo erra",
            "Os pontos de teste no plano da projeção 2D do ecrã Preparar. Zonas com muitos "
            "erros mostram onde o modelo tem mais dificuldade (muitas vezes, onde as classes "
            "se misturam).",
        ),
        "cluster_sizes": Explanation(
            "O tamanho dos grupos",
            "Quantos pontos de teste ficaram em cada grupo. Grupos muito desequilibrados (um "
            "enorme e vários minúsculos) costumam querer dizer um número de grupos errado.",
        ),
        "dendrogram": Explanation(
            "Dendrograma",
            "Cada «U» é uma junção de dois grupos, à altura da distância a que se deu (só as "
            "últimas 30). A linha horizontal é o corte que dá os grupos escolhidos: uma junção "
            "muito mais alta do que as de baixo é um bom sítio para cortar.",
        ),
        "explained_variance": Explanation(
            "Variância explicada",
            "Quanto da variância dos dados guarda cada componente, por ordem. A soma é o que a "
            "redução guardou; o que falta perdeu-se nela.",
        ),
        "anomaly_scores": Explanation(
            "As pontuações de anomalia",
            "A pontuação de cada ponto de teste. Acima do limiar (a linha), o detetor marca o "
            "ponto como anomalia. Com uma coluna de referência, as cores são as anomalias "
            "verdadeiras: as que ficam abaixo do limiar escaparam ao detetor, e os pontos "
            "normais acima dele são falsos alarmes. Sem ela, as cores são o que o detetor "
            "decidiu.",
        ),
        "anomaly_roc": Explanation(
            "A ROC da pontuação",
            "A curva ROC da pontuação contra as anomalias verdadeiras (da coluna de "
            "referência): não depende do limiar.",
        ),
        # Como ler os gráficos do ecrã Explicar.
        "importance_bars": Explanation(
            "Como ler a importância",
            "Cada barra é quanto a métrica principal piora no teste quando se baralha a "
            "feature, e o traço à volta da ponta é o desvio entre as repetições. Perto de zero, "
            "ou negativa: o modelo não precisa dessa feature, ou outra correlacionada "
            "substitui-a.",
        ),
        "dependence_curves": Explanation(
            "Como ler a dependência parcial",
            "A linha forte é a previsão média quando todas as linhas do teste recebem esse "
            "valor da feature; as ténues são algumas linhas, uma a uma. Os traços em baixo "
            "mostram onde há dados: longe deles, a curva é pura extrapolação. Os valores da "
            "feature estão na escala depois da preparação (padronizados, se for o caso).",
        ),
        # Como ler o que um modelo aprendeu (a página de cada modelo e o fim de um treino).
        "impurity_importance": Explanation(
            "Importância pelas impurezas",
            "Quanto cada feature reduziu a impureza nas divisões das árvores, somado em todas "
            "elas e normalizado para dar 1. Favorece as features com muitos valores "
            "diferentes: a importância por permutação (ecrã Explicar) é mais honesta.",
        ),
        "tree_diagram": Explanation(
            "Como ler a árvore",
            "Cada nó faz uma pergunta a uma feature («x2 ≤ 0,26?»): os pontos que respondem "
            "sim seguem para a esquerda, os outros para a direita. As folhas, em baixo, dão a "
            "previsão: a classe mais frequente dos pontos de treino que lá chegam, com a parte "
            "que ela tem. Uma folha pura (100%) só tem pontos de uma classe; muitas folhas "
            "pequenas e puras são o sinal de uma árvore que decorou o treino.",
        ),
    }
)
