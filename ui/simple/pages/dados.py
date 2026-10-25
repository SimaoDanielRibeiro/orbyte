"""Ecrã 1 · Dados: carregar um ficheiro, escolher um exemplo ou gerar dados sintéticos."""

from collections.abc import Callable

import streamlit as st

from components.forms import schema_form
from components.layout import render_header, render_why
from components.notices import render_error, render_notices
from components.state import current_dataset, get_lab, select_dataset
from components.tables import show_preview
from ml_lab.core.services import DatasetSummary, MLLabError, NoticeLevel, explain
from ml_lab.formatting import format_count

_UPLOAD_TYPES = ["csv", "tsv", "txt", "gz", "xlsx", "xlsm"]


def main() -> None:
    """Desenha o ecrã."""
    render_header("dados")
    render_why("input", "validation")
    samples_tab, file_tab, synthetic_tab = st.tabs(["Exemplos", "Ficheiro", "Sintéticos"])
    with samples_tab:
        _samples()
    with file_tab:
        _upload()
    with synthetic_tab:
        _synthetic()

    dataset = current_dataset()
    st.divider()
    if dataset is None:
        st.info(
            "Escolhe um exemplo, carrega um ficheiro ou gera dados sintéticos para começar.",
            icon=":material/arrow_upward:",
        )
        return
    _switcher(dataset)
    _show(dataset)


def _load(action: Callable[[], DatasetSummary]) -> None:
    try:
        summary = action()
    except MLLabError as error:
        render_error(error)
        return
    select_dataset(summary)


def _samples() -> None:
    datasets = get_lab().datasets
    samples = {sample.name: sample for sample in datasets.samples()}
    name = st.selectbox(
        "Dataset de exemplo",
        list(samples),
        format_func=lambda option: samples[option].title,
        key="dados.sample",
        persist_state="session",
    )
    sample = samples[name]
    st.markdown(sample.description)
    st.caption(
        f"{format_count(sample.n_rows)} linhas · {sample.n_columns} colunas · alvo habitual: "
        f"{sample.target} ({sample.task_type.label.lower()}) · fonte: {sample.source}"
    )
    if st.button("Carregar exemplo", type="primary", icon=":material/download:", key="dados.load"):
        _load(lambda: datasets.load_sample(name))


def _upload() -> None:
    uploaded = st.file_uploader("Ficheiro CSV ou Excel", type=_UPLOAD_TYPES, key="dados.upload")
    st.caption(
        "O separador (vírgula, ponto e vírgula, tabulação) e a codificação são detetados sozinhos. "
        "O ficheiro fica guardado em data/uploads/, que não vai para o Git."
    )
    if uploaded is not None and st.button(
        "Carregar ficheiro", type="primary", icon=":material/upload_file:", key="dados.load_file"
    ):
        _load(lambda: get_lab().datasets.upload(uploaded.name, uploaded.getvalue()))


def _synthetic() -> None:
    datasets = get_lab().datasets
    generators = {generator.name: generator for generator in datasets.generators()}
    name = st.selectbox(
        "Gerador",
        list(generators),
        format_func=lambda option: generators[option].title,
        key="dados.generator",
        persist_state="session",
    )
    generator = generators[name]
    st.markdown(generator.description)
    parameters = schema_form(generator.parameters_schema, key=f"dados.generator.{name}")
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        step=1,
        help="A mesma seed dá sempre os mesmos pontos.",
        key="dados.generator_seed",
        persist_state="session",
    )
    if st.button(
        "Gerar dados", type="primary", icon=":material/scatter_plot:", key="dados.generate"
    ):
        _load(lambda: datasets.generate(name, parameters, int(seed)))


def _switcher(current: DatasetSummary) -> None:
    loaded = {dataset.id: dataset for dataset in get_lab().datasets.loaded()}
    if len(loaded) < 2:
        return
    options = list(loaded)
    choice = st.selectbox(
        "Datasets carregados nesta sessão",
        options,
        index=options.index(current.id),
        format_func=lambda option: loaded[option].name,
    )
    if choice != current.id:
        select_dataset(loaded[choice])
        st.rerun()


def _show(dataset: DatasetSummary) -> None:
    st.subheader(dataset.name)
    if dataset.description:
        st.caption(dataset.description)
    report = dataset.validation
    rows, columns, warnings = st.columns(3)
    rows.metric("Linhas", format_count(dataset.n_rows))
    columns.metric("Colunas", format_count(dataset.n_columns))
    warnings.metric(
        "Avisos da validação",
        sum(1 for notice in report.notices if notice.level is not NoticeLevel.INFO),
    )
    render_notices(report.notices, notes_title="Notas da leitura")
    if not report.is_usable:
        st.error("Estes dados não podem seguir para as próximas etapas: corrige os erros acima.")
        return
    if not report.notices:
        st.success("Os dados passaram a validação sem avisos.", icon=":material/check_circle:")

    st.markdown("**Pré-visualização**")
    show_preview(get_lab().datasets.preview(dataset.id), key="dados.preview")
    st.markdown("**Colunas e tipos**")
    st.dataframe(
        {
            "Coluna": [column.name for column in dataset.columns],
            "Tipo no pandas": [column.dtype for column in dataset.columns],
            "Tipo inferido": [column.kind.label for column in dataset.columns],
            "Porquê": [column.kind_reason for column in dataset.columns],
        },
        hide_index=True,
        width="stretch",
    )
    _next_step()


def _next_step() -> None:
    """Os dois caminhos depois dos dados: os agentes (automático) ou passo a passo (manual)."""
    choice = explain("automatic_or_manual")
    st.markdown(f"**{choice.title}**")
    st.caption(choice.text)
    automatic, manual = st.columns(2)
    with automatic.container(border=True):
        st.markdown("**Modo automático**")
        st.write("Dizes o que queres prever, e os agentes procuram o melhor modelo.")
        st.page_link("pages/agentes.py", label="Pedir aos agentes", icon=":material/groups:")
    with manual.container(border=True):
        st.markdown("**Modo manual**")
        st.write("Exploras, preparas e escolhes o modelo tu, a ver o treino por dentro.")
        st.page_link(
            "pages/explorar.py", label="Explorar este dataset", icon=":material/arrow_forward:"
        )


main()
