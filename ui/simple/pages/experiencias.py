"""Ecrã 8 · Experiências: o histórico dos runs, a comparação lado a lado e a reprodução.

Cada treino que termina fica registado sozinho; guardar junta-lhe o modelo treinado e o
treino passo a passo (a animação e, numa rede, as camadas). Aqui filtra-se o histórico,
comparam-se de 2 a 5 runs e abre-se, reproduz-se, renomeia-se ou apaga-se um run.
"""

from collections.abc import Sequence
from datetime import datetime

import streamlit as st

from charts.history import loss_comparison_chart, metrics_comparison_chart
from components.layout import render_header, render_why
from components.notices import render_error, render_notices
from components.state import get_lab, save_job, show_result, training_in_progress
from ml_lab.core.services import MLLabError, RunComparison, RunEntry, TaskType
from ml_lab.formatting import format_number, format_percent

# O que aparece nos filtros quando não se filtra nada.
_ALL = "Todos"


def main() -> None:
    """Desenha o ecrã."""
    render_header("experiencias")
    everything = get_lab().history.runs()
    if not everything:
        st.info(
            "Ainda não há runs no histórico: cada treino que termina fica registado aqui sozinho.",
            icon=":material/history:",
        )
        st.page_link(
            "pages/modelo.py", label="Ir para o ecrã Modelo", icon=":material/arrow_forward:"
        )
        render_why("experiment_tracking")
        return
    entries = _filtered(everything)
    _table(entries)
    st.subheader("Comparar runs")
    _compare(entries)
    st.subheader("Um run")
    _one_run(entries)
    render_why("experiment_tracking", "comparing_runs", "reproducibility")


def _filtered(everything: Sequence[RunEntry]) -> tuple[RunEntry, ...]:
    tasks: list[TaskType | None] = [
        None,
        *sorted({entry.task_type for entry in everything}, key=lambda task: task.label),
    ]
    datasets: list[str | None] = [None, *sorted({entry.dataset_name for entry in everything})]
    names = {entry.model_name: entry.model_display_name for entry in everything}
    models: list[str | None] = [None, *sorted(names, key=names.__getitem__)]
    task_column, dataset_column, model_column = st.columns(3)
    task = task_column.selectbox(
        "Tarefa",
        tasks,
        format_func=lambda option: _ALL if option is None else option.label,
        key="experiencias.task",
    )
    dataset = dataset_column.selectbox(
        "Dataset",
        datasets,
        format_func=lambda option: _ALL if option is None else option,
        key="experiencias.dataset",
    )
    model = model_column.selectbox(
        "Modelo",
        models,
        format_func=lambda option: _ALL if option is None else names[option],
        key="experiencias.model",
    )
    return get_lab().history.runs(task_type=task, dataset_name=dataset, model_name=model)


def _table(entries: Sequence[RunEntry]) -> None:
    if not entries:
        st.info("Nenhum run corresponde a estes filtros.", icon=":material/filter_alt_off:")
        return
    st.dataframe(
        {
            "Quando": [_when(entry.created_at) for entry in entries],
            "Nome": [entry.name for entry in entries],
            "Dataset": [entry.dataset_name for entry in entries],
            "Tarefa": [entry.task_type.label for entry in entries],
            "Métrica principal": [_primary(entry) for entry in entries],
            "Guardado": ["sim" if entry.saved else "—" for entry in entries],
            "Seed": [entry.seed for entry in entries],
            "Run": [entry.run_id for entry in entries],
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"{len(entries)} runs, do mais recente para o mais antigo. Os guardados voltam a abrir "
        "com o modelo e o treino passo a passo; os outros, com tudo menos isso."
    )


def _compare(entries: Sequence[RunEntry]) -> None:
    labels = {entry.run_id: _label(entry) for entry in entries}
    chosen = st.multiselect(
        "Runs a comparar (de 2 a 5)",
        list(labels),
        format_func=labels.__getitem__,
        max_selections=5,
        key="experiencias.compare",
    )
    if len(chosen) < 2:
        st.caption(
            "Escolhe pelo menos dois runs para os ver lado a lado. Para aprender com a "
            "comparação, escolhe runs que mudem uma coisa de cada vez."
        )
        return
    try:
        comparison = get_lab().history.compare(chosen)
    except MLLabError as error:
        render_error(error)
        return
    render_notices(comparison.notices, notes_title="Leitura da comparação")
    _metrics_table(comparison)
    if any(metric.as_percent for metric in comparison.metrics):
        st.plotly_chart(metrics_comparison_chart(comparison), width="stretch")
    if comparison.differences:
        st.markdown("**O que muda entre os runs**")
        table: dict[str, list[str]] = {
            "Escolha": [difference.label for difference in comparison.differences]
        }
        for index, label in enumerate(comparison.labels):
            table[label] = [difference.values[index] for difference in comparison.differences]
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.caption("Os runs têm exatamente a mesma configuração e a mesma seed.")
    if comparison.losses:
        st.markdown("**A loss ao longo do treino**")
        st.plotly_chart(loss_comparison_chart(comparison), width="stretch")


def _metrics_table(comparison: RunComparison) -> None:
    table: dict[str, list[str]] = {
        "Métrica": [metric.display_name for metric in comparison.metrics]
    }
    for index, label in enumerate(comparison.labels):
        table[label] = [
            _written(metric.values[index], metric.as_percent)
            + (" ★" if metric.best == index else "")
            for metric in comparison.metrics
        ]
    st.markdown(f"**Métricas no {comparison.partition.label.lower()}** (★ o melhor)")
    st.dataframe(table, hide_index=True, width="stretch")


def _one_run(entries: Sequence[RunEntry]) -> None:
    if not entries:
        return
    labels = {entry.run_id: _label(entry) for entry in entries}
    run_id = st.selectbox(
        "Run", list(labels), format_func=labels.__getitem__, key="experiencias.run"
    )
    entry = next(entry for entry in entries if entry.run_id == run_id)
    kept = "guardado, com o modelo e o treino passo a passo" if entry.saved else "só registado"
    st.caption(
        f"{entry.model_display_name} em {entry.dataset_name} · "
        f"{entry.task_type.label.lower()} · seed {entry.seed} · {kept}"
    )
    busy = training_in_progress()
    if busy:
        st.info(
            "Há um treino a decorrer: espera que acabe para abrir ou reproduzir outro run.",
            icon=":material/hourglass_top:",
        )
    open_column, reproduce_column, save_column = st.columns(3)
    if open_column.button(
        "Abrir",
        icon=":material/open_in_new:",
        disabled=busy,
        width="stretch",
        key="experiencias.open",
        help="Mostra este run nos ecrãs Treinar, Avaliar e Explicar.",
    ):
        _open(entry.run_id)
    if reproduce_column.button(
        "Reproduzir",
        icon=":material/replay:",
        disabled=busy,
        width="stretch",
        key="experiencias.reproduce",
        help="Treina outra vez com a mesma configuração e a mesma seed; o run novo entra no "
        "histórico, para os comparares.",
    ):
        _reproduce(entry.run_id)
    if save_column.button(
        "Guardar",
        icon=":material/save:",
        disabled=entry.saved,
        width="stretch",
        key="experiencias.save",
        help="Junta o modelo treinado e o treino passo a passo (só nos runs desta sessão, que "
        "estão em memória).",
    ):
        _save(entry.run_id)
    with st.expander("Mudar o nome ou apagar", icon=":material/edit:"):
        _rename_or_delete(entry)


def _open(run_id: str) -> None:
    try:
        result = get_lab().history.open(run_id)
    except MLLabError as error:
        render_error(error)
        return
    show_result(result)
    st.switch_page("pages/avaliar.py")


def _reproduce(run_id: str) -> None:
    try:
        job = get_lab().history.reproduce(run_id)
    except MLLabError as error:
        render_error(error)
        return
    save_job(job)
    st.switch_page("pages/treinar.py")


def _save(run_id: str) -> None:
    try:
        get_lab().history.save(run_id)
    except MLLabError as error:
        render_error(error)
        return
    st.rerun()


def _rename_or_delete(entry: RunEntry) -> None:
    name = st.text_input("Nome", value=entry.name, key=f"experiencias.name.{entry.run_id}")
    if st.button("Mudar o nome", key="experiencias.rename"):
        try:
            get_lab().history.rename(entry.run_id, name)
        except MLLabError as error:
            render_error(error)
        else:
            st.rerun()
    confirmed = st.checkbox(
        "Quero apagar este run (não se pode desfazer)",
        key=f"experiencias.confirm.{entry.run_id}",
    )
    if st.button(
        "Apagar", icon=":material/delete:", disabled=not confirmed, key="experiencias.delete"
    ):
        get_lab().history.delete(entry.run_id)
        st.rerun()


def _label(entry: RunEntry) -> str:
    return f"{entry.name} · {_when(entry.created_at)} · {entry.run_id}"


def _when(moment: datetime) -> str:
    """O momento na hora local, por extenso curto ("24/10/2026 15:30")."""
    return moment.astimezone().strftime("%d/%m/%Y %H:%M")


def _primary(entry: RunEntry) -> str:
    value = entry.primary_value
    if entry.primary_label is None or value is None:
        return "—"
    return f"{entry.primary_label}: {_written(value, entry.primary_as_percent)}"


def _written(value: float | None, as_percent: bool) -> str:
    if value is None:
        return "—"
    return format_percent(value) if as_percent else format_number(value, 3)


main()
