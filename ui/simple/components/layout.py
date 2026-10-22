"""Elementos comuns a todos os ecrãs da interface simples."""

import streamlit as st

from components.screens import SCREENS, Screen
from ml_lab import __version__

_EXPERIMENT_FIELDS = ("Dataset", "Tarefa", "Modelo", "Último resultado")


def render_sidebar() -> None:
    """Barra lateral com o estado da experiência atual (vazio até à fase 1)."""
    with st.sidebar:
        st.subheader("Experiência atual")
        for field in _EXPERIMENT_FIELDS:
            st.markdown(f"**{field}:** —")
        st.button(
            "Guardar experiência",
            icon=":material/save:",
            disabled=True,
            width="stretch",
            help="Disponível a partir da fase 5 (Memória).",
        )
        st.caption(f"ML Lab {__version__}")


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
