"""Catálogo dos ecrãs da interface, pela ordem natural de uma experiência.

A lista é a mesma nas duas versões da interface (simples e aprimorada); só muda a
apresentação.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Screen:
    """Um ecrã da interface.

    Attributes:
        number: Posição na barra de navegação.
        slug: Nome do ficheiro em ``pages/`` e do caminho no URL.
        title: Título do ecrã.
        icon: Ícone Material do Streamlit.
        phases: Fases do backend que dão conteúdo ao ecrã.
        does: O que o utilizador faz neste ecrã.
        sees: O que o utilizador vê neste ecrã.
    """

    number: int
    slug: str
    title: str
    icon: str
    phases: tuple[int, ...]
    does: str
    sees: str

    @property
    def page_path(self) -> str:
        """Caminho do ficheiro do ecrã, relativo a ``app.py``."""
        return f"pages/{self.slug}.py"

    @property
    def arrives(self) -> str:
        """Quando o ecrã ganha conteúdo, por exemplo "na fase 3" ou "nas fases 2 e 4"."""
        numbers = [str(phase) for phase in self.phases]
        if len(numbers) == 1:
            return f"na fase {numbers[0]}"
        return f"nas fases {', '.join(numbers[:-1])} e {numbers[-1]}"


SCREENS: tuple[Screen, ...] = (
    Screen(
        number=1,
        slug="dados",
        title="Dados",
        icon=":material/database:",
        phases=(1,),
        does=(
            "Carregar um CSV ou Excel, escolher um dataset de exemplo ou gerar um sintético, "
            "com controlos de ruído e de número de pontos."
        ),
        sees="Lista de datasets, pré-visualização das primeiras linhas e erros de validação.",
    ),
    Screen(
        number=2,
        slug="explorar",
        title="Explorar",
        icon=":material/query_stats:",
        phases=(1,),
        does="Navegar pelas colunas do dataset.",
        sees=(
            "Perfil de cada coluna, histogramas, mapa de correlações, mapa de nulos, "
            "desequilíbrio de classes e avisos."
        ),
    ),
    Screen(
        number=3,
        slug="preparar",
        title="Preparar",
        icon=":material/tune:",
        phases=(1, 2),
        does=(
            "Escolher a coluna-alvo, confirmar o tipo de tarefa, aplicar limpezas e "
            "transformações e definir a divisão em treino e teste."
        ),
        sees=(
            "O antes e depois de cada operação, o histórico de passos (linhagem) e o "
            "tamanho do treino e do teste."
        ),
    ),
    Screen(
        number=4,
        slug="modelo",
        title="Modelo",
        icon=":material/account_tree:",
        phases=(2, 4),
        does=(
            "Escolher um algoritmo do catálogo, filtrado pela tarefa, e ajustar os hiperparâmetros."
        ),
        sees="Cartões por família, a ficha didática e o formulário gerado da ficha técnica.",
    ),
    Screen(
        number=5,
        slug="treinar",
        title="Treinar",
        icon=":material/model_training:",
        phases=(3,),
        does="Iniciar, pausar e cancelar o treino, e percorrê-lo passo a passo.",
        sees=(
            "Barra de progresso, loss em tempo real e a animação dos snapshots: fronteira de "
            "decisão, centróides ou árvore."
        ),
    ),
    Screen(
        number=6,
        slug="avaliar",
        title="Avaliar",
        icon=":material/analytics:",
        phases=(2, 4),
        does="Ler os resultados do modelo.",
        sees=(
            "Cartões de métricas com explicação, matriz de confusão, curvas ROC e PR, resíduos, "
            "silhouette, comparação treino vs. teste e diagnósticos."
        ),
    ),
    Screen(
        number=7,
        slug="explicar",
        title="Explicar",
        icon=":material/lightbulb:",
        phases=(4,),
        does="Escolher uma previsão ou o modelo inteiro.",
        sees="Importância das features e gráficos de dependência parcial.",
    ),
    Screen(
        number=8,
        slug="experiencias",
        title="Experiências",
        icon=":material/history:",
        phases=(5,),
        does="Filtrar o histórico, escolher 2 ou 3 runs e reproduzir um run.",
        sees="Tabela de runs e comparação lado a lado de métricas e gráficos.",
    ),
    Screen(
        number=9,
        slug="laboratorio",
        title="Laboratório",
        icon=":material/science:",
        phases=(4,),
        does="Mexer num controlo e ver o efeito imediato.",
        sees=(
            "Demonstrações de conceitos: overfitting, taxa de aprendizagem, o k do KNN e "
            "regularização."
        ),
    ),
    Screen(
        number=10,
        slug="reinforcement",
        title="Reinforcement learning",
        icon=":material/smart_toy:",
        phases=(8,),
        does="Escolher o ambiente e o agente e iniciar o treino.",
        sees="Recompensa por episódio, tabela Q e a política aprendida.",
    ),
)

_BY_SLUG = {screen.slug: screen for screen in SCREENS}


def get_screen(slug: str) -> Screen:
    """Devolve o ecrã com o ``slug`` dado (o nome do seu ficheiro em ``pages/``)."""
    return _BY_SLUG[slug]
