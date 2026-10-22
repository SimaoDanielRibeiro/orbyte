"""Interface simples do ML Lab: entrada da app e navegação entre ecrãs.

Abrir com ``uv run streamlit run ui/simple/app.py``.

O Streamlit volta a correr este script a cada interação. A navegação fica no topo,
da esquerda para a direita pela ordem de uma experiência, e a barra lateral mostra
o estado da experiência atual em todos os ecrãs.
"""

import streamlit as st

from components.layout import render_sidebar
from components.screens import SCREENS
from ml_lab.config import get_settings
from ml_lab.logging import configure_logging


@st.cache_resource
def _bootstrap() -> None:
    """Prepara o backend uma única vez por processo: logs e pastas de trabalho."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    settings.ensure_directories()


st.set_page_config(page_title="ML Lab", page_icon=":material/neurology:", layout="wide")
_bootstrap()

navigation = st.navigation(
    [st.Page(screen.page_path, title=screen.title, icon=screen.icon) for screen in SCREENS],
    position="top",
)
render_sidebar()
navigation.run()
