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
    }
)
