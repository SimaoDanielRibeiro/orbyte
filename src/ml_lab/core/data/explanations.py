"""Explicações didáticas de cada etapa da camada de dados.

São os textos do painel "Porquê?" da interface. Vivem no backend para as duas
interfaces (a simples e a aprimorada) mostrarem exatamente o mesmo, escrito uma
só vez.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from ml_lab.core.foundation.errors import ComponentNotFoundError


@dataclass(frozen=True, slots=True)
class Explanation:
    """Um tema explicado.

    Attributes:
        title: O título.
        text: A explicação, em Markdown.
    """

    title: str
    text: str


EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "input": Explanation(
            "Entrada",
            "Os dados chegam como uma tabela: cada linha é um exemplo (um passageiro, uma "
            "flor) e cada coluna uma característica. Podem vir de um ficheiro teu, de um "
            "dataset de exemplo ou de um gerador de dados sintéticos, em que controlas a forma "
            "dos dados e o ruído. A tabela original nunca é alterada: cada etapa trabalha "
            "numa cópia.",
        ),
        "validation": Explanation(
            "Validação",
            "Antes de mais nada, confirma-se se os dados são utilizáveis: se o ficheiro tem "
            "linhas e colunas, se o cabeçalho não tem nomes repetidos, se a codificação dos "
            "acentos foi bem detetada. A validação não altera nada; só avisa.",
        ),
        "profiling": Explanation(
            "Perfil",
            "O perfil descreve cada coluna: o tipo (o que a coluna *significa*, não só como "
            "está guardada), quantos valores faltam, quantos são distintos, as estatísticas e a "
            "distribuição. Serve para decidir o que limpar e como transformar. Em datasets "
            "grandes, as distribuições usam uma amostra; as contagens usam todas as linhas.",
        ),
        "cleaning": Explanation(
            "Limpeza",
            "A limpeza corrige problemas que não dependem de nenhuma amostra: remover "
            "duplicados, converter tipos, retirar identificadores e colunas inúteis. Por não "
            "aprender nada com os dados, pode acontecer antes da divisão. Cada operação fica "
            "registada na linhagem, com o antes e o depois.",
        ),
        "task": Explanation(
            "Tarefa",
            "Escolher o alvo é escolher a pergunta: o que queres prever? Um alvo com classes "
            "(texto, booleano, inteiro com poucos valores) pede classificação; um número "
            "contínuo pede regressão; sem alvo, só há tarefas não supervisionadas. O sistema "
            "sugere, mas és tu que decides.",
        ),
        "split": Explanation(
            "Divisão",
            "O modelo aprende com o **treino** e é avaliado no **teste**, em linhas que nunca "
            "viu: é a única forma honesta de saber se generaliza ou se só decorou. A divisão "
            "**estratificada** mantém a proporção das classes em todas as partes (as do alvo "
            "ou, nas tarefas sem alvo, as da coluna de referência); a "
            "**temporal** treina com o passado e testa com o futuro. A seed torna a divisão "
            "repetível.",
        ),
        "transform": Explanation(
            "Transformação",
            "Os modelos precisam de números sem buracos e em escalas parecidas. A imputação "
            "preenche os nulos, a codificação troca categorias por colunas 0/1 e a escala põe "
            "as colunas na mesma ordem de grandeza. Tudo isto **aprende só com o treino** (a "
            "mediana, as categorias, a média e o desvio) e aplica-se depois à validação e ao "
            "teste.",
        ),
        "leakage": Explanation(
            "Fuga de dados (data leakage)",
            "Há fuga de dados quando o modelo, durante o treino, recebe informação a que não "
            "teria acesso no momento de prever: por exemplo, uma média calculada com o teste, "
            "ou uma coluna que só existe depois de se saber a resposta. O resultado no teste "
            "fica melhor do que a realidade. Aprender as transformações só com o treino evita "
            "o caso mais comum.",
        ),
        "projection": Explanation(
            "Projeção 2D",
            "Um gráfico só mostra duas dimensões. Com duas features, usam-se as próprias; com "
            "mais, a PCA encontra as duas direções em que os dados variam mais e projeta-os "
            "nelas. É uma cópia só para ver: o modelo usa todas as features.",
        ),
        "output": Explanation(
            "Saída",
            "No fim, a tabela vira matrizes NumPy de números decimais (float64), e as classes "
            "do alvo viram números 0, 1, 2… É aqui que acaba o pandas: daqui para a frente, os "
            "modelos só veem números, mas os nomes das features e das classes seguem com eles.",
        ),
    }
)


def explain(topic: str) -> Explanation:
    """A explicação de um tema (``"split"``, ``"transform"``...).

    Raises:
        ComponentNotFoundError: Se o tema não existir.
    """
    try:
        return EXPLANATIONS[topic]
    except KeyError:
        raise ComponentNotFoundError("explanations", topic, EXPLANATIONS) from None
