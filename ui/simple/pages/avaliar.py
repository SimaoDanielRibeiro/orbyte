"""Ecrã 6 · Avaliar: métricas, diagnósticos e comparação treino vs. teste."""

from typing import Any

import streamlit as st

from charts.evaluation import confusion_matrix_chart, train_test_chart
from components.layout import render_header
from components.notices import render_notices
from components.state import last_result
from ml_lab.core.services import ExperimentResult, Partition
from ml_lab.formatting import format_number

# Quantas métricas aparecem em destaque, no topo. Num ecrã, um texto solto (mesmo uma
# docstring de constante) é desenhado pela "magia" do Streamlit: daí o comentário.
_CARDS = 4


def main() -> None:
    """Desenha o ecrã."""
    render_header("avaliar")
    result = last_result()
    if result is None:
        st.info("Ainda não treinaste nenhum modelo.", icon=":material/model_training:")
        st.page_link(
            "pages/modelo.py", label="Ir para o ecrã Modelo", icon=":material/arrow_forward:"
        )
        return
    st.markdown(
        f"**{result.model_display_name}** em **{result.dataset_name}** · "
        f"{result.task_type.label.lower()} · seed {result.seed} · "
        f"{format_number(result.duration_seconds, 2)} s · run `{result.run_id}`"
    )
    test = result.reports[Partition.TEST]
    train = result.reports[Partition.TRAIN]
    st.caption("No teste, com linhas que o modelo nunca viu:")
    columns = st.columns(_CARDS)
    for column, metric in zip(columns, test.metrics[:_CARDS], strict=False):
        column.metric(metric.display_name, metric.formatted, help=metric.what, border=True)
    render_notices([*result.notices, *test.notices], notes_title="Leitura dos resultados")

    metrics_tab, confusion_tab, compare_tab, learned_tab = st.tabs(
        ["Métricas", "Matriz de confusão", "Treino vs. teste", "O que o modelo aprendeu"]
    )
    with metrics_tab:
        for metric in test.metrics:
            with st.expander(f"{metric.display_name}: {metric.formatted}"):
                st.markdown(f"**O que mede.** {metric.what}")
                st.markdown(f"**Como se lê.** {metric.how_to_read}")
                st.caption(f"No treino: {train.metric(metric.name).formatted}.")
    with confusion_tab:
        _confusion(result)
    with compare_tab:
        st.plotly_chart(train_test_chart(train, test), width="stretch")
        table: dict[str, list[str]] = {"Métrica": [m.display_name for m in test.metrics]}
        for part, report in result.reports.items():
            table[part.label] = [report.metric(m.name).formatted for m in test.metrics]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "O modelo aprendeu com o treino; o teste mostra como se sai com dados que nunca viu. "
            "Uma diferença grande a favor do treino é o sinal de overfitting."
        )
    with learned_tab:
        _learned(result)
    with st.expander("Configuração do run (para o repetir)", icon=":material/settings:"):
        st.json(result.config.model_dump(mode="json"), expanded=False)


def _confusion(result: ExperimentResult) -> None:
    parts = list(result.reports)
    part = st.segmented_control(
        "Parte dos dados",
        parts,
        default=Partition.TEST,
        required=True,
        format_func=lambda option: option.label,
        key="avaliar.partition",
    )
    report = result.reports[part or Partition.TEST]
    if report.confusion_matrix is None:
        st.info("Este tipo de tarefa não tem matriz de confusão.")
        return
    st.plotly_chart(confusion_matrix_chart(report.confusion_matrix), width="stretch")
    st.caption(
        "Cada linha é a classe real e cada coluna a prevista: a diagonal são os acertos, o resto "
        "são as confusões do modelo."
    )


def _learned(result: ExperimentResult) -> None:
    state = result.model_state
    coefficients: list[list[float]] | None = state.get("coefficients")
    if not coefficients:
        st.info("Este modelo não mostra o que aprendeu.")
        return
    labels = result.class_labels
    # Na classificação binária há um só vetor de pesos: o da classe positiva.
    names = [f"«{labels[-1]}»"] if len(coefficients) == 1 else [f"«{label}»" for label in labels]
    table: dict[str, Any] = {"Feature": list(result.feature_names)}
    for name, weights in zip(names, coefficients, strict=True):
        table[f"Peso para {name}"] = [format_number(weight, 3) for weight in weights]
    st.dataframe(table, hide_index=True, width="stretch")
    intercepts = ", ".join(format_number(value, 3) for value in state.get("intercept", []))
    st.caption(
        f"Ordenada (b): {intercepts}. Iterações do otimizador: {state.get('iterations', '—')}. "
        "Um peso positivo empurra para a classe, um negativo afasta. Com as features "
        "padronizadas, os pesos comparam-se entre si: os maiores (em valor absoluto) são as "
        "features que mais pesam na decisão."
    )


main()
