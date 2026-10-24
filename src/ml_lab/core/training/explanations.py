"""Explicações didáticas do treino: os textos do painel "Porquê?" do ecrã Treinar.

Como as da camada de dados, vivem no backend para as duas interfaces mostrarem o
mesmo texto, escrito uma só vez.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

TRAINING_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "loss": Explanation(
            "Loss",
            "A loss (função de perda) mede quanto o modelo erra no treino, num só número: a "
            "log-loss na classificação, o erro quadrático médio na regressão. Treinar é "
            "procurar os parâmetros que a tornam o mais pequena possível. Uma loss que desce "
            "depressa e depois estabiliza é um treino saudável; uma que sobe, ou salta de um "
            "lado para o outro, pede uma taxa de aprendizagem menor.",
        ),
        "gradient_descent": Explanation(
            "Gradiente descendente",
            "O gradiente diz em que direção a loss mais sobe. O gradiente descendente dá "
            "passos na direção contrária, do tamanho da **taxa de aprendizagem**: como descer "
            "uma encosta com nevoeiro, sempre pelo lado mais inclinado. Cada passagem por "
            "todos os exemplos de treino é uma **época**. Passos pequenos demais arrastam o "
            "treino; grandes demais saltam por cima do fundo e a loss explode.",
        ),
        "snapshots": Explanation(
            "Snapshots",
            "Um snapshot é uma fotografia do modelo a meio do treino: os pesos e as previsões "
            "numa grelha que cobre os dados. Guardar todos os passos ocuparia memória a mais, "
            "por isso guardam-se no máximo 60, espalhados pelo treino (o primeiro e o último "
            "ficam sempre). Rever os snapshots por ordem é ver o modelo a aprender.",
        ),
        "decision_boundary": Explanation(
            "Fronteira de decisão",
            "A fronteira de decisão é a linha onde o modelo muda de opinião: de um lado prevê "
            "uma classe, do outro a outra (na regressão logística, onde a probabilidade é "
            "50%). Desenha-se no plano da projeção 2D do ecrã Preparar; as direções que o "
            "plano não mostra ficam na média dos dados de treino. Com mais de duas features, "
            "é por isso um corte da fronteira verdadeira, e alguns pontos podem parecer do "
            "lado errado sem o estarem.",
        ),
        "neural_network": Explanation(
            "Rede neuronal",
            "Uma rede neuronal densa é uma sequência de camadas. Cada neurónio faz uma soma "
            "pesada das entradas, mais uma ordenada, e passa-a por uma **ativação** não linear "
            "(ReLU, tanh, sigmoide). Sem a ativação, várias camadas seguidas eram só uma conta "
            "linear, e a fronteira seria sempre uma reta; com ela, a rede dobra o espaço e "
            "desenha fronteiras curvas. Treinar é ajustar todos os pesos ao mesmo tempo para a "
            "loss descer.",
        ),
        "backpropagation": Explanation(
            "Retropropagação",
            "Para dar um passo, a rede precisa do gradiente da loss em cada peso. A "
            "**retropropagação** calcula-os todos de uma vez, da saída para a entrada, com a "
            "regra da cadeia: o gradiente de uma camada usa o da camada seguinte. O treino "
            "percorre os dados em **lotes**: em cada lote, previsões, loss, gradientes e um "
            "passo do otimizador (o Adam ajusta o passo de cada peso sozinho).",
        ),
        "early_stopping": Explanation(
            "Early stopping",
            "Uma parte do treino fica de lado, a **validação**, e a rede nunca aprende com ela. "
            "Enquanto a loss na validação desce, a rede está a aprender o que generaliza; "
            "quando deixa de descer e a do treino continua a cair, começa a decorar. O treino "
            "pára depois de algumas épocas sem melhorar (a **paciência**) e a rede fica com os "
            "pesos da melhor época.",
        ),
        "network_diagram": Explanation(
            "As camadas da rede",
            "Cada círculo é um neurónio, da entrada (à esquerda) à saída (à direita). Nas "
            "camadas escondidas, a cor é a ativação média do neurónio nos exemplos de treino. "
            "Um neurónio **parado** (cinzento) dá sempre o mesmo valor, seja qual for o "
            "exemplo, e por isso não ajuda a distinguir nada. Muda o passo para ver as camadas "
            "a organizarem-se ao longo do treino.",
        ),
        "network_gradients": Explanation(
            "Os gradientes de cada camada",
            "O tamanho do gradiente dos pesos de cada camada, época a época (em escala "
            "logarítmica). Gradientes que encolhem muito de camada para camada, até à primeira, "
            "são o **gradiente que desaparece**: as primeiras camadas quase não aprendem. "
            "Acontece com a sigmoide em redes fundas, e é uma das razões por que a ReLU é a "
            "ativação habitual.",
        ),
        "network_stuck": Explanation(
            "Neurónios parados",
            "A proporção de neurónios de cada camada que dão sempre o mesmo valor. Na ReLU, um "
            "neurónio cuja soma ficou sempre negativa dá sempre zero e o gradiente que passa "
            "por ele é zero: morreu, e não volta. Na tanh e na sigmoide, um neurónio saturado "
            "(sempre perto dos extremos) quase não aprende. Muitos parados pedem uma taxa de "
            "aprendizagem menor.",
        ),
        "network_activations": Explanation(
            "Ativações",
            "A distribuição das saídas dos neurónios de cada camada, num passo do treino. Na "
            "ReLU, uma barra alta em zero são os neurónios desligados para esses exemplos; na "
            "tanh e na sigmoide, barras altas nos extremos são neurónios saturados.",
        ),
    }
)
