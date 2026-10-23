"""Elementos comuns a todos os ecrãs da interface simples."""

import streamlit as st

from components.screens import SCREENS, Screen, get_screen
from components.state import current_dataset, current_preparation
from ml_lab import __version__
from ml_lab.core.services import DatasetSummary, explain
from ml_lab.formatting import format_count


def render_sidebar() -> None:
    """Barra lateral com o estado da experiência atual: dataset, tarefa, modelo e resultado."""
    dataset = current_dataset()
    preparation = current_preparation()
    fields = {
        "Dataset": (
            f"{dataset.name} ({format_count(dataset.n_rows)} × {dataset.n_columns})"
            if dataset
            else "—"
        ),
        "Tarefa": preparation.summary.task_type.label if preparation else "—",
        "Modelo": "—",
        "Último resultado": "—",
    }
    with st.sidebar:
        st.subheader("Experiência atual")
        for field, value in fields.items():
            st.markdown(f"**{field}:** {value}")
        st.button(
            "Guardar experiência",
            icon=":material/save:",
            disabled=True,
            width="stretch",
            help="Disponível a partir da fase 5 (Memória).",
        )
        st.caption(f"ML Lab {__version__}")


def render_header(slug: str) -> Screen:
    """Título e posição do ecrã; devolve o ecrã, para quem precisar dos seus textos."""
    screen = get_screen(slug)
    st.title(screen.title, icon=screen.icon)
    st.caption(f"Ecrã {screen.number} de {len(SCREENS)} · {screen.does}")
    return screen


def render_why(*topics: str) -> None:
    """Painel "Porquê?" com as explicações didáticas dos temas (os textos vêm do backend)."""
    with st.expander("Porquê?", icon=":material/school:"):
        for topic in topics:
            explanation = explain(topic)
            st.markdown(f"**{explanation.title}.** {explanation.text}")


def require_dataset() -> DatasetSummary | None:
    """O dataset atual; sem ele, explica o que fazer (e o ecrã deve terminar)."""
    dataset = current_dataset()
    if dataset is None:
        st.info("Ainda não há nenhum dataset carregado.", icon=":material/database:")
        st.page_link(
            "pages/dados.py", label="Ir para o ecrã Dados", icon=":material/arrow_forward:"
        )
    return dataset


def render_placeholder(screen: Screen) -> None:
    """Ecrã ainda vazio: mostra o que vai ter e em que fase ganha conteúdo."""
    st.title(screen.title, icon=screen.icon)
    st.caption(f"Ecrã {screen.number} de {len(SCREENS)}")
    st.info(f"Ecrã ainda vazio: ganha conteúdo {screen.arrives}.", icon=":material/construction:")
    does, sees = st.columns(2)
    with does:
        st.markdown("**O que vais fazer**")
        st.write(screen.does)
    with sees:
        st.markdown("**O que vais ver**")
        st.write(screen.sees)
