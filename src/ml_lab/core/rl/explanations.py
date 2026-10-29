"""Textos do painel "Porquê?" do ecrã Reinforcement learning (só se escrevem aqui)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

RL_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "reinforcement_learning": Explanation(
            "Reinforcement learning",
            "Não há dataset: um **agente** aprende a agir num **ambiente**. Em cada passo vê o "
            "estado em que está, escolhe uma ação e recebe uma **recompensa**. Ninguém lhe diz "
            "qual era a ação certa; só no fim percebe o que valeu a pena. Com muitos "
            "**episódios** (tentativas, do início ao fim), aprende que ações levam a mais "
            "recompensa: é aprender por tentativa e erro.",
        ),
        "rl_state": Explanation(
            "O estado",
            "O agente vê onde está: a célula da grelha. Não vê o mapa nem sabe onde fica o "
            "objetivo; sabe só o que já aprendeu sobre cada célula, os seus valores Q.",
        ),
        "rl_action": Explanation(
            "A ação",
            "Escolhe para onde ir. Com probabilidade ε, escolhe ao acaso: **explora**. No resto "
            "das vezes, escolhe a ação de maior Q: **aproveita** o que já sabe. O ε começa em 1 "
            "e desce ao longo do treino.",
        ),
        "rl_reward": Explanation(
            "A recompensa",
            "O ambiente responde com a célula onde o agente foi parar e uma recompensa: −0,01 "
            "por cada passo, −1 numa armadilha e +1 no objetivo. As duas últimas acabam o "
            "episódio.",
        ),
        "rl_update": Explanation(
            "A correção",
            "O agente compara o que esperava, Q(s, a), com o que viu: a recompensa mais o "
            "melhor Q da célula seguinte, descontado por γ. A diferença é a **surpresa**, e o "
            "Q anda uma fração α dela. Na grelha, a cor da célula muda: é assim que o valor do "
            "objetivo se espalha para trás.",
        ),
        "q_values": Explanation(
            "Valores Q e política",
            "O **Q(estado, ação)** é a recompensa que o agente espera ganhar até ao fim do "
            "episódio se fizer aquela ação ali, e depois jogar o melhor que sabe. A "
            "**política** é a regra de escolha: em cada estado, a ação de maior Q. No "
            "início, os Q são todos zero e o agente não sabe nada; o objetivo é acertá-los.",
        ),
        "temporal_difference": Explanation(
            "O erro de diferença temporal",
            "Depois de cada passo, o Q-learning compara o que esperava, Q(s, a), com o que viu: "
            "a recompensa mais o melhor Q do estado seguinte, descontado. A diferença é a "
            "**surpresa**, e o Q anda uma fração dela (a taxa de aprendizagem α). É assim que o "
            "+1 do objetivo se espalha para trás, de célula em célula, até à partida.",
        ),
        "exploration": Explanation(
            "Explorar ou aproveitar (ε-greedy)",
            "Se o agente escolhesse sempre a ação de maior Q, nunca descobria caminhos "
            "melhores do que o primeiro que encontrou. Com **ε-greedy**, escolhe ao acaso com "
            "probabilidade ε (explorar) e a melhor que conhece no resto das vezes "
            "(aproveitar). O ε começa alto, quando não se sabe nada, e desce ao longo do "
            "treino.",
        ),
        "discount": Explanation(
            "O desconto (γ)",
            "Uma recompensa futura vale menos do que uma já: cada passo para o futuro "
            "multiplica-a por γ. Com γ = 0,95, chegar ao objetivo daqui a 10 passos vale 0,95¹⁰ "
            "≈ 0,6. É por isso que o caminho curto vale mais do que o longo, e que um γ baixo "
            "faz um agente míope, que só liga ao que está perto.",
        ),
        "deep_q_network": Explanation(
            "DQN: uma rede no lugar da tabela",
            "O DQN aprende os mesmos Q, mas com uma rede neuronal, o que permite estados "
            "demasiados para uma tabela. Treinar uma rede com a surpresa de cada passo é "
            "instável, e por isso há duas peças: a **memória de repetição**, de onde se tiram "
            "lotes de passos ao acaso (passos seguidos são parecidos demais), e a **rede-alvo**, "
            "uma cópia que só se atualiza de vez em quando e que calcula os alvos (sem ela, o "
            "alvo mexia-se a cada passo de treino).",
        ),
        "policy_map": Explanation(
            "Como ler a grelha",
            "A cor de cada célula é o seu valor: o maior Q das suas ações, ou seja, quanto o "
            "agente espera ganhar a partir dali. As setas são a política: a ação que escolhe "
            "em cada célula. A linha é o caminho que faria a partir da partida, sem escorregar. "
            "Percorre os episódios para ver o valor a espalhar-se a partir do objetivo.",
        ),
        "reward_curve": Explanation(
            "A recompensa por episódio",
            "Cada ponto é a recompensa de um episódio do treino; a linha é a média móvel. "
            "Os episódios são ruidosos porque o agente ainda explora ao acaso. A curva sobe "
            "quando o agente aprende a chegar ao objetivo e a fazê-lo em menos passos.",
        ),
        "success_curve": Explanation(
            "Os episódios que chegam ao objetivo",
            "A parte dos últimos episódios (a mesma janela da média móvel) em que o agente "
            "chegou ao objetivo. Sobe quando ele aprende o caminho; a treinar, raramente chega "
            "aos 100%, porque o agente continua a explorar ao acaso de vez em quando.",
        ),
        "q_table": Explanation(
            "Como ler a tabela Q",
            "Uma grelha por ação: a cor de cada célula é o Q dessa ação ali, o que o agente "
            "espera ganhar se a fizer e depois jogar o melhor que sabe. Em cada célula, a "
            "política escolhe a ação com o Q mais alto das quatro grelhas.",
        ),
        "rl_evaluation": Explanation(
            "Treino e teste num agente",
            "No **treino**, o agente ainda explora: as métricas são dos últimos episódios, "
            "com ε. No **teste**, joga episódios novos com a política aprendida, sem explorar "
            "e sem aprender: é o que sabe fazer. A diferença entre os dois é o preço de "
            "explorar. No penhasco, cai nas armadilhas a treinar, mas atravessa no teste.",
        ),
    }
)
"""Os temas do ecrã Reinforcement learning, pelo nome."""
