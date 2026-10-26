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
            "mais, a PCA encontra, só com o treino, as duas direções em que os dados variam "
            "mais e projeta nelas todas as partes. É uma cópia só para ver: o modelo usa todas "
            "as features.",
        ),
        "output": Explanation(
            "Saída",
            "No fim, a tabela vira matrizes NumPy de números decimais (float64), e as classes "
            "do alvo viram números 0, 1, 2… É aqui que acaba o pandas: daqui para a frente, os "
            "modelos só veem números, mas os nomes das features e das classes seguem com eles.",
        ),
        # Como ler cada gráfico e cada tabela dos ecrãs de dados: as interfaces mostram-nos
        # por baixo deles, e o painel "Porquê?" também os pode juntar.
        "upload": Explanation(
            "Carregar um ficheiro",
            "Aceita CSV (separado por vírgulas, por ponto e vírgula ou por tabulações, também "
            "comprimido em `.gz`) e Excel (`.xlsx`, `.xlsm`). O separador e a codificação dos "
            "acentos são detetados sozinhos. O ficheiro fica guardado na pasta dos uploads "
            "(`data/uploads/`), que não vai para o Git.",
        ),
        "correlation_matrix": Explanation(
            "Matriz de correlação",
            "Cada célula é a correlação de Pearson entre duas colunas numéricas: 1 é uma relação "
            "linear perfeita, -1 também, mas ao contrário (uma sobe quando a outra desce), e 0 "
            "é nenhuma relação linear. Duas features muito correlacionadas dizem quase o mesmo "
            "ao modelo. Correlação não é causalidade.",
        ),
        "missing_map": Explanation(
            "Mapa dos valores em falta",
            "Cada linha do mapa é uma linha do dataset (uma seleção espalhada por ele todo) e "
            "cada coluna é uma coluna; as células marcadas são valores em falta. Faixas na "
            "vertical são colunas com muitos nulos; na horizontal, linhas quase vazias. Nulos "
            "que aparecem juntos em várias colunas costumam ter a mesma causa.",
        ),
        "class_balance": Explanation(
            "Equilíbrio das classes",
            "Quantas linhas tem cada classe do alvo. O equilíbrio é a parte que a classe menos "
            "frequente tem da mais frequente: 100% é perfeitamente equilibrado. Com classes "
            "muito desequilibradas, um modelo que responde sempre a classe maior parece bom na "
            "exatidão; a divisão estratificada e métricas como o F1 passam a contar mais.",
        ),
        "group_means": Explanation(
            "Médias por classe",
            "A média de cada feature numérica em cada classe do alvo. Diferenças grandes entre "
            "as classes indicam features que ajudam a separá-las; médias quase iguais, features "
            "que sozinhas dizem pouco.",
        ),
        "lineage": Explanation(
            "Linhagem",
            "Cada linha é uma operação da preparação, pela ordem em que correu, com o que mudou: "
            "as linhas antes e depois e as colunas que entraram, saíram ou mudaram. É com esta "
            "lista que a preparação se repete, igual, noutro dia ou com dados novos.",
        ),
        "split_classes": Explanation(
            "Classes em cada parte",
            "A proporção de cada classe no treino, na validação e no teste. Na divisão "
            "estratificada, as percentagens são quase iguais em todas as partes; na aleatória, "
            "podem afastar-se, sobretudo com classes raras e poucos dados.",
        ),
        "pca_loadings": Explanation(
            "Pesos das componentes",
            "Cada componente principal é uma soma das features (já transformadas), cada uma "
            "com o seu peso. Um peso grande, positivo ou negativo, diz que a feature conta "
            "muito nessa direção; perto de zero, que quase não conta.",
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
