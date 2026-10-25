"""Ecrã 8 · Explicar: o que pesa nas previsões do modelo, no conjunto e numa previsão."""

import streamlit as st

from charts.evaluation import contributions_chart, dependence_chart, importance_chart
from components.layout import render_header, render_why
from components.notices import render_error
from components.state import get_lab, last_result
from ml_lab.core.services import ExperimentResult, MLLabError
from ml_lab.formatting import format_number, format_percent


def main() -> None:
    """Desenha o ecrã."""
    render_header("explicar")
    result = last_result()
    if result is None:
        st.info("Ainda não treinaste nenhum modelo.", icon=":material/model_training:")
        st.page_link(
            "pages/modelo.py", label="Ir para o ecrã Modelo", icon=":material/arrow_forward:"
        )
        return
    if not result.task_type.requires_target:
        st.info(
            f"A explicabilidade mostra o que pesa nas previsões, e "
            f"{result.task_type.label.lower()} não prevê um alvo. Para perceber o que o modelo "
            "encontrou, vê os centros, os grupos ou o mapa no ecrã Avaliar.",
            icon=":material/lightbulb:",
        )
        return
    st.markdown(
        f"**{result.model_display_name}** em **{result.dataset_name}** · "
        f"{result.task_type.label.lower()} · run `{result.run_id}`"
    )
    importance, dependence, row = st.tabs(
        ["Importância das features", "Dependência parcial", "Uma previsão"]
    )
    with importance:
        _importance(result)
    with dependence:
        _dependence(result)
    with row:
        _row(result)
    render_why("permutation_importance", "partial_dependence", "local_explanation")


def _importance(result: ExperimentResult) -> None:
    key = f"explicar.importance.{result.run_id}"
    st.markdown(
        "Para cada feature, baralham-se os seus valores no teste e mede-se quanto o modelo "
        "piora. Repete-se 5 vezes por feature, por isso pode demorar com muitas features."
    )
    if st.button("Calcular a importância", icon=":material/play_arrow:", key=f"{key}.run"):
        st.session_state[key] = True
    if not st.session_state.get(key):
        return
    try:
        with st.spinner("A baralhar cada feature…"):
            importance = get_lab().analysis.permutation_importance(result.run_id)
    except MLLabError as error:
        render_error(error)
        return
    st.plotly_chart(importance_chart(importance), width="stretch")
    baseline = (
        format_percent(importance.baseline)
        if importance.as_percent
        else format_number(importance.baseline, 3)
    )
    st.caption(
        f"Sem baralhar nada, {importance.metric_label} no teste é {baseline}. Cada barra é "
        "quanto piora com a feature baralhada (a linha é o desvio entre repetições). Perto de "
        "zero, ou negativa: o modelo não precisa dessa feature, ou outra correlacionada "
        "substitui-a."
    )


def _dependence(result: ExperimentResult) -> None:
    feature = st.selectbox(
        "Feature",
        list(result.feature_names),
        key=f"explicar.feature.{result.run_id}",
        help="Os valores estão na escala depois da preparação (padronizados, se for o caso).",
    )
    try:
        dependence = get_lab().analysis.partial_dependence(result.run_id, feature)
    except MLLabError as error:
        render_error(error)
        return
    st.plotly_chart(dependence_chart(dependence), width="stretch")
    st.caption(
        "A linha grossa é a previsão média quando todas as linhas do teste recebem esse valor "
        "da feature; as finas são algumas linhas, uma a uma. Os traços em baixo mostram onde "
        "há dados: longe deles, a curva é pura extrapolação."
    )


def _row(result: ExperimentResult) -> None:
    analysis = get_lab().analysis
    try:
        rows = analysis.test_rows(result.run_id)
    except MLLabError as error:
        render_error(error)
        return
    row = st.number_input(
        "Linha do teste",
        min_value=0,
        max_value=rows - 1,
        value=0,
        key=f"explicar.row.{result.run_id}",
        help=f"O teste tem {rows} linhas.",
    )
    try:
        explanation = analysis.explain_row(result.run_id, int(row))
    except MLLabError as error:
        render_error(error)
        return
    columns = st.columns(3)
    columns[0].metric("Previsão", explanation.prediction, border=True)
    columns[1].metric("Valor real", explanation.actual or "—", border=True)
    if explanation.probabilities:
        best = max(explanation.probabilities, key=lambda pair: pair[1])
        columns[2].metric(f"Probabilidade de «{best[0]}»", format_percent(best[1]), border=True)
    st.plotly_chart(contributions_chart(explanation), width="stretch")
    if explanation.method == "weights":
        base = "—" if explanation.base is None else format_number(explanation.base, 3)
        st.caption(
            f"Modelo linear: cada barra é peso × valor da feature, e a previsão (ou a pontuação "
            f"da classe) é a ordenada ({base}) mais a soma das barras. Verde empurra para cima, "
            "vermelho para baixo."
        )
    else:
        st.caption(
            "Cada barra é quanto a previsão muda por a feature ter este valor, em vez da média do "
            "treino. É uma aproximação: ignora as interações entre features."
        )


main()
