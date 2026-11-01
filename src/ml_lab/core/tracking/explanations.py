"""Textos do painel "Porquê?" dos ecrãs Histórico e Guardar.

Registar, comparar e reproduzir runs; e guardar um modelo num pacote, levá-lo e voltar a
carregá-lo (ADR 0039).
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

TRACKING_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "experiment_tracking": Explanation(
            "Registo de experiências",
            "Cada treino que termina fica registado sozinho: a configuração, a seed, as "
            "métricas, a curva da loss e o que o modelo aprendeu. Chega para comparar runs e "
            "para os repetir. **Guardar** um run junta-lhe o modelo treinado e o treino passo a "
            "passo (a animação e, numa rede neuronal, as camadas em cada snapshot), que pesam "
            "mais: um run guardado volta a abrir com tudo, mesmo depois de fechar a app.",
        ),
        "comparing_runs": Explanation(
            "Comparar runs",
            "Uma comparação só ensina alguma coisa se os runs diferirem numa escolha de cada "
            "vez: o mesmo dataset, a mesma preparação e a mesma divisão, e só um "
            "hiperparâmetro diferente. Se mudarem duas coisas ao mesmo tempo, não se sabe qual "
            "delas fez a diferença. Por isso, a comparação mostra só as escolhas que mudam.",
        ),
        "run_history": Explanation(
            "Como ler o histórico",
            "Os runs vêm do mais recente para o mais antigo, com a métrica principal de cada um "
            "na parte que o resume (o teste, quase sempre). Os guardados voltam a abrir com o "
            "modelo e o treino passo a passo; os outros, com tudo menos isso: as métricas, a "
            "configuração e o que o modelo aprendeu, mas sem os gráficos que precisam do modelo "
            "nem o ecrã Explicar.",
        ),
        "reproducibility": Explanation(
            "Reprodutibilidade",
            "Um resultado só conta se se puder repetir. No ML Lab, a seed raiz decide todas as "
            "outras (a da divisão, a do modelo, a dos folds), por isso a mesma configuração com "
            "a mesma seed dá exatamente o mesmo modelo e as mesmas métricas. **Reproduzir** um "
            "run treina-o outra vez assim; se as métricas mudarem, alguma coisa não é "
            "determinística, ou os dados mudaram.",
        ),
        "model_package": Explanation(
            "O pacote do modelo",
            "Um modelo sozinho não chega para prever: ele aprendeu com os dados **depois** da "
            "preparação. Uma linha nova (uma pessoa com 22 anos, do sexo feminino) tem de "
            "passar pelas mesmas transformações, com os valores aprendidos no treino: a mesma "
            "mediana para preencher a idade em falta, as mesmas categorias, a mesma média e o "
            "mesmo desvio para escalar. Por isso o pacote leva a preparação com o modelo. Se a "
            "preparação mudasse entre o treino e o uso, o modelo recebia números que nunca viu "
            "e enganava-se sem avisar.",
        ),
        "model_card": Explanation(
            "A ficha do modelo",
            "Quem recebe um modelo precisa de saber, antes de confiar nele: o que prevê e a "
            "partir de quê, com que dados aprendeu, como se portou no teste (e comparado com o "
            "palpite mais simples), e onde não se deve usar. O limite mais importante são os "
            "intervalos que ele viu: fora deles, a previsão é uma **extrapolação**, um palpite "
            "sem dados por trás. A ficha é o README.md do pacote, escrito para pessoas.",
        ),
        "pickle_safety": Explanation(
            "Porque é que abrir um modelo pede confirmação",
            "O modelo vai num ficheiro joblib, que é um *pickle* do Python: abri-lo volta a "
            "criar os objetos que lá estão, e isso pode correr código. Um pacote feito neste "
            "ML Lab traz a assinatura dele e abre logo; um feito noutro sítio pode trazer código "
            "de quem o fez. Por isso a ficha, a avaliação e a preparação (que são texto) "
            "leem-se logo, e o modelo só abre depois de confirmares que confias em quem te deu "
            "o ficheiro. Um pacote alterado depois de feito nunca abre o modelo.",
        ),
        "onnx_format": Explanation(
            "O ONNX",
            "O ONNX (Open Neural Network Exchange) é um formato aberto para trocar modelos "
            "treinados entre programas. Descreve o modelo como um grafo de operações simples "
            "(somar, multiplicar, comparar), que o ONNX Runtime corre em Python, C#, Java, "
            "JavaScript e noutras linguagens, sem o ML Lab nem o scikit-learn. O `model.onnx` "
            "do pacote junta a preparação e o modelo: recebe as colunas como chegam e devolve "
            "a previsão. Antes de entrar no pacote, corre com as linhas de exemplo e tem de "
            "dar as mesmas previsões do laboratório; se não der, fica de fora. Não é um "
            "pickle, por isso não corre código de ninguém.",
        ),
        "memorized_rows": Explanation(
            "Modelos que guardam os dados",
            "Alguns modelos não resumem os dados em pesos: guardam-nos. O KNN guarda todas as "
            "linhas do treino (é com elas que compara cada ponto novo), uma SVM guarda os "
            "vetores de suporte (as linhas na fronteira) e o DBSCAN os pontos centrais. Quem "
            "tiver um destes modelos consegue ver essas linhas, já transformadas. Antes de "
            "partilhar um modelo treinado com dados pessoais, pensa em quem o vai ter.",
        ),
        "package_versions": Explanation(
            "Porque se registam as versões",
            "O manifesto diz com que versões do Python, do NumPy, do scikit-learn e do PyTorch "
            "se fez o pacote. Um pickle depende delas: noutra versão, o modelo pode não abrir, "
            "ou prever de outra maneira sem dar erro. Com as versões à vista, sabe-se o que "
            "instalar para o repetir; é a reprodutibilidade estendida para lá da seed.",
        ),
    }
)
"""Os temas do registo de experiências e dos pacotes."""
