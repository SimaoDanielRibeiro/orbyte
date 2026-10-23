"""Avisos didáticos: erros e avisos à vista, notas num painel recolhido."""

from collections.abc import Sequence

import streamlit as st

from ml_lab.core.services import Notice, NoticeLevel

_ICONS = {
    NoticeLevel.ERROR: ":material/error:",
    NoticeLevel.WARNING: ":material/warning:",
    NoticeLevel.INFO: ":material/info:",
}


def render_notices(notices: Sequence[Notice], *, notes_title: str = "Notas") -> None:
    """Mostra os avisos: erros e avisos destacados; as notas (``INFO``) numa lista recolhida."""
    for notice in notices:
        if notice.level is NoticeLevel.ERROR:
            st.error(notice.message, icon=_ICONS[notice.level])
    for notice in notices:
        if notice.level is NoticeLevel.WARNING:
            st.warning(notice.message, icon=_ICONS[notice.level])
    notes = [notice for notice in notices if notice.level is NoticeLevel.INFO]
    if notes:
        with st.expander(f"{notes_title} ({len(notes)})", icon=_ICONS[NoticeLevel.INFO]):
            for notice in notes:
                st.markdown(f"- {notice.message}")


def render_error(error: Exception) -> None:
    """Mostra um erro do ML Lab (as mensagens já explicam o que fazer)."""
    st.error(str(error), icon=_ICONS[NoticeLevel.ERROR])
