"""Elementos comuns a todos os ecrãs da interface simples."""

import streamlit as st

from components.screens import SCREENS, Screen, get_screen
from components.state import (
    chosen_model,
    current_dataset,
    current_preparation,
    get_lab,
    last_result,
)
from ml_lab import __version__
from ml_lab.core.services import DatasetSummary, ExperimentResult, MLLabError, explain
from ml_lab.formatting import format_count


def render_sidebar() -> None:
    """Barra lateral com o estado da experiência atual: dataset, tarefa, modelo e resultado."""
    dataset = current_dataset()
    preparation = current_preparation()
    model = chosen_model()
    result = last_result()
    primary = result.primary if result else None
    where = result.main_partition.label.lower() if result else ""
    fields = {
        "Dataset": (
            f"{dataset.name} ({format_count(dataset.n_rows)} × {dataset.n_columns})"
            if dataset
            else "—"
        ),
        "Tarefa": preparation.summary.task_type.label if preparation else "—",
        "Modelo": model.display_name if model else "—",
        "Último resultado": (
            f"{primary.display_name} no {where}: {primary.formatted}" if primary else "—"
        ),
    }
    with st.sidebar:
        st.subheader("Experiência atual")
        for field, value in fields.items():
            st.markdown(f"**{field}:** {value}")
        _save_button(result)
        st.caption(f"ML Lab {__version__}")


def _save_button(result: ExperimentResult | None) -> None:
    """Guardar experiência: junta ao histórico o modelo e o treino passo a passo do último run."""
    known = saved = False
    if result is not None:
        try:
            saved = get_lab().history.entry(result.run_id).saved
            known = True
        except MLLabError:  # o run não ficou no histórico (por exemplo, foi apagado)
            pass
    st.button(
        "Experiência guardada" if saved else "Guardar experiência",
        icon=":material/check:" if saved else ":material/save:",
        disabled=not known or saved,
        width="stretch",
        on_click=_save,
        args=(None if result is None else result.run_id,),
        key="sidebar.save",
        help=(
            "Cada run já fica no histórico (ecrã Experiências). Guardar junta-lhe o modelo "
            "treinado e o treino passo a passo (a animação e, numa rede, as camadas): volta a "
            "abrir com tudo, mesmo depois de fechar a app."
        ),
    )


def _save(run_id: str | None) -> None:
    if run_id is None:
        return
    try:
        get_lab().history.save(run_id)
    except MLLabError as error:
        st.toast(str(error), icon=":material/error:")
    else:
        st.toast("Experiência guardada: está no ecrã Experiências.", icon=":material/save:")


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
