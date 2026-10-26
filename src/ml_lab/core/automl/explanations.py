"""Textos do painel "Porquê?" do ecrã Agentes (só se escrevem aqui)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

AUTOML_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "automl": Explanation(
            "Os agentes (AutoML)",
            "No modo automático, três tipos de agentes fazem o trabalho de um analista: o "
            "**agente de dados** prepara a tabela, um **agente por família de modelos** "
            "propõe candidatos (as árvores, os lineares, as redes…) e o **coordenador** "
            "distribui o orçamento e escolhe o vencedor. Cada decisão fica na conversa, com a "
            "razão. Não há magia nem um modelo de linguagem: são regras e medições, e com a "
            "mesma seed a procura repete-se tal e qual.",
        ),
        "automatic_or_manual": Explanation(
            "Automático ou manual?",
            "O modo automático responde depressa a «que modelo serve melhor para isto?» e "
            "mostra porquê. O manual (Explorar, Preparar, Modelo, Treinar) é onde se aprende a "
            "fazer cada escolha e se vê o treino por dentro. Os dois acabam no mesmo sítio: o "
            "vencedor dos agentes é um run como os outros, que se avalia e explica da mesma "
            "maneira, e qualquer candidato do quadro se pode treinar à mão.",
        ),
        "model_selection": Explanation(
            "Escolher sem olhar para o teste",
            "Os agentes comparam os candidatos com **validação cruzada**, só no treino: o "
            "treino parte-se em folds, e cada candidato treina com uns e é avaliado no que "
            "ficou de fora. O teste fica guardado até ao fim e só o vencedor o vê, uma vez. Se "
            "se escolhesse pelo teste, o teste passava a ser mais um treino: o vencedor seria "
            "o que teve mais sorte nele, e a sua nota deixava de dizer como se porta em dados "
            "novos.",
        ),
        "screening_and_tuning": Explanation(
            "Triagem e afinação",
            "Afinar todos os modelos custaria muito. Por isso há duas rondas: na **triagem**, "
            "cada modelo é experimentado uma vez, com os valores por omissão (uma primeira "
            "impressão barata); na **afinação**, só os melhores (os finalistas) recebem o resto "
            "do **orçamento**, para os seus agentes experimentarem outras combinações de "
            "hiperparâmetros. O orçamento conta candidatos, e cada candidato treina uma vez por "
            "fold.",
        ),
        "random_search": Explanation(
            "Procura aleatória",
            "Na afinação, os agentes não percorrem uma grelha com todas as combinações: "
            "sorteiam-nas, dentro do espaço de procura de cada modelo (por exemplo, o C do SVM "
            "entre 0,1 e 100, em escala logarítmica). Com o mesmo número de tentativas, a "
            "procura aleatória experimenta mais valores diferentes de cada hiperparâmetro, e "
            "isso conta porque, em cada problema, só alguns deles importam a sério.",
        ),
        "winners_curse": Explanation(
            "O vencedor é um pouco otimista",
            "Entre muitos candidatos parecidos, o que ganha a validação cruzada costuma ser bom "
            "e, também, ter tido um pouco de sorte nos folds. Por isso a sua nota no teste "
            "costuma ficar um pouco abaixo da da validação cruzada, e quanto mais candidatos, "
            "mais. Diferenças menores do que o desvio padrão entre folds são, na prática, "
            "empates: nesses casos, o modelo mais simples é uma escolha tão boa como o vencedor.",
        ),
        "train_candidate": Explanation(
            "Treinar um candidato à mão",
            "Qualquer candidato do quadro treina como um run normal, com a mesma preparação e a "
            "mesma seed: vês o treino passo a passo e avalia-lo no teste. É a ponte do modo "
            "automático para o manual, por exemplo para comparar o vencedor com um modelo mais "
            "simples que ficou quase empatado.",
        ),
        "data_agent": Explanation(
            "O que o agente de dados decide",
            "O agente de dados olha para o perfil (não para os valores das linhas) e decide "
            "como no ecrã Preparar: remove identificadores, texto livre e colunas quase vazias, "
            "confirma a tarefa, divide com 20% para o teste (estratificado, se houver classes), "
            "preenche os nulos, passa as categorias a one-hot e padroniza os números. Com "
            "classes desequilibradas, sugere otimizar a exatidão equilibrada. Se já preparaste "
            "os dados à mão, usa a tua preparação.",
        ),
    }
)
