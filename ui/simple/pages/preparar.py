"""Ecrã 4 · Preparar: alvo, tipo de tarefa, limpeza, transformações e divisão.

A preparação volta a correr sempre que um controlo muda, para se ver logo o efeito
de cada escolha na linhagem, nos tamanhos e na projeção.
"""

from collections.abc import Callable
from functools import partial

import streamlit as st

from charts.projection import projection_chart
from components.layout import render_header, render_why, require_dataset
from components.notices import render_error, render_notices
from components.state import Preparation, clear_preparation, get_lab, save_preparation
from components.tables import lineage_table, show_preview
from ml_lab.core.services import (
    CleaningStep,
    ColumnKind,
    DatasetSummary,
    ImputationStrategy,
    MLLabError,
    PreparationConfig,
    PreparationSummary,
    ScalingMethod,
    SplitConfig,
    SplitStrategy,
    TaskType,
    TransformConfig,
)
from ml_lab.formatting import format_count, format_number, format_percent


def main() -> None:
    """Desenha o ecrã."""
    render_header("preparar")
    dataset = require_dataset()
    if dataset is None:
        return
    render_why("cleaning", "task", "split", "transform", "leakage", "projection", "output")
    key = f"preparar.{dataset.id}"

    st.subheader("1. Alvo e tarefa")
    target, task_type = _task(dataset, key)
    if task_type is None:
        clear_preparation()
        return
    # O que a tarefa muda (a coluna de referência, as divisões possíveis) vem do backend.
    options = get_lab().datasets.task_options(task_type, target)
    if options.reference_note:
        st.info(options.reference_note, icon=":material/compare_arrows:")
    st.subheader("2. Limpeza")
    steps, dropped = _cleaning(dataset, key, target)
    st.subheader("3. Divisão")
    split, seed = _split(dataset, key, options.split_strategies)
    st.subheader("4. Transformações")
    transforms = _transforms(dataset, key, target, dropped)

    st.subheader("5. Resultado")
    try:
        config = PreparationConfig(
            target=target,
            task_type=task_type,
            cleaning=tuple(steps),
            split=split(),
            transforms=transforms(),
        )
        summary = get_lab().datasets.prepare(dataset.id, config, seed)
    except MLLabError as error:
        render_error(error)
        clear_preparation()
        return
    save_preparation(Preparation(dataset.id, config, seed, summary))
    _result(summary)


def _task(dataset: DatasetSummary, key: str) -> tuple[str | None, TaskType | None]:
    options: list[str | None] = [None, *(column.name for column in dataset.columns)]
    suggested = dataset.suggested_target
    target = st.selectbox(
        "Coluna-alvo: o que queres prever?",
        options,
        index=options.index(suggested) if suggested in options else 0,
        format_func=lambda option: (
            "Sem alvo (aprendizagem não supervisionada)" if option is None else option
        ),
        key=f"{key}.target",
        persist_state="session",
    )
    suggestion = get_lab().datasets.suggest_task(dataset.id, target)
    st.markdown(f"**Sugestão:** {suggestion.reason}")
    render_notices(suggestion.notices)
    if suggestion.task_type is None:
        return target, None
    task_type = st.selectbox(
        "Tipo de tarefa",
        list(suggestion.allowed),
        format_func=lambda option: option.label,
        key=f"{key}.task.{target}",
        persist_state="session",
        help="A sugestão vem primeiro; podes escolher outra das que fazem sentido para este alvo.",
    )
    return target, task_type


def _cleaning(
    dataset: DatasetSummary, key: str, target: str | None
) -> tuple[list[CleaningStep], list[str]]:
    suggestions = get_lab().datasets.suggest_cleaning(dataset.id, target)
    steps: list[CleaningStep] = []
    for suggestion in suggestions:
        operation = suggestion.step.operation
        if operation == "drop_duplicates":
            label = f"Remover linhas duplicadas ({suggestion.reason[:-1].lower()})"
        elif operation == "convert_type":
            label = f"Converter {suggestion.column}: {suggestion.reason}"
        else:
            continue  # as colunas a remover têm um controlo próprio, mais abaixo
        toggle_key = f"{key}.{operation}.{suggestion.column}"
        if st.toggle(label, value=True, key=toggle_key, persist_state="session"):
            steps.append(suggestion.step)
    if not any(s.step.operation == "drop_duplicates" for s in suggestions):
        st.caption("Não há linhas duplicadas.")

    reasons = {s.column: s.reason for s in suggestions if s.step.operation == "drop_columns"}
    candidates = [column.name for column in dataset.columns if column.name != target]
    dropped = st.multiselect(
        "Remover colunas",
        candidates,
        default=[column for column in reasons if column in candidates],
        key=f"{key}.drop.{target}",
        persist_state="session",
        help="As sugeridas não ajudam o modelo; podes juntar ou tirar colunas.",
    )
    for column in dropped:
        if column in reasons:
            st.caption(f"• {reasons[column]}")
    if dropped:
        steps.append(CleaningStep("drop_columns", {"columns": list(dropped)}))

    remaining = [column for column in dataset.columns if column.name not in dropped]
    with st.expander("Mais operações de limpeza"):
        missing = st.multiselect(
            "Remover as linhas com valores em falta em",
            [column.name for column in remaining],
            key=f"{key}.drop_missing",
            persist_state="session",
        )
        if missing:
            steps.append(CleaningStep("drop_missing_rows", {"columns": list(missing)}))
        numeric = [
            column.name
            for column in remaining
            if column.kind is ColumnKind.NUMERIC and column.name != target
        ]
        extremes = st.multiselect(
            "Tratar os valores extremos (regra do IQR) em",
            numeric,
            key=f"{key}.outliers",
            persist_state="session",
        )
        method = st.radio(
            "Como tratar os extremos",
            ["clip", "remove"],
            format_func=lambda option: (
                "Limitar aos limites" if option == "clip" else "Remover as linhas"
            ),
            horizontal=True,
            key=f"{key}.outliers_method",
            persist_state="session",
        )
        steps.extend(
            CleaningStep("treat_outliers", {"column": column, "method": method})
            for column in extremes
        )
    return steps, list(dropped)


def _split(
    dataset: DatasetSummary, key: str, strategies: tuple[SplitStrategy, ...]
) -> tuple[Callable[[], SplitConfig], int]:
    """Os controlos da divisão; devolve como criar a configuração (pode ser inválida) e a seed."""
    strategy = st.segmented_control(
        "Estratégia",
        strategies,
        default=strategies[0],
        required=True,
        format_func=lambda option: option.label,
        # Uma chave por conjunto de opções: a escolha guardada é sempre uma das oferecidas.
        key=f"{key}.strategy.{'-'.join(strategies)}",
        persist_state="session",
    )
    test, validation, seed_column = st.columns(3)
    test_size = test.slider(
        "Teste", 0.05, 0.5, 0.2, 0.05, key=f"{key}.test", persist_state="session"
    )
    validation_size = validation.slider(
        "Validação", 0.0, 0.3, 0.0, 0.05, key=f"{key}.validation", persist_state="session"
    )
    seed = seed_column.number_input(
        "Seed",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        step=1,
        key=f"{key}.seed",
        persist_state="session",
        help="A mesma seed dá sempre a mesma divisão.",
    )
    time_column = None
    if strategy == SplitStrategy.TIME:
        times = [
            column.name
            for column in dataset.columns
            if column.kind in (ColumnKind.DATETIME, ColumnKind.NUMERIC)
        ]
        time_column = st.selectbox(
            "Coluna do tempo",
            [None, *times],
            format_func=lambda option: "A ordem das linhas" if option is None else option,
            key=f"{key}.time_column",
            persist_state="session",
        )
    # O slider devolve, por exemplo, 0.30000000000000004; arredondar evita linhas a mais.
    return partial(
        SplitConfig,
        strategy=strategy or strategies[0],
        test_size=round(test_size, 2),
        validation_size=round(validation_size, 2),
        time_column=time_column,
    ), int(seed)


def _transforms(
    dataset: DatasetSummary, key: str, target: str | None, dropped: list[str]
) -> Callable[[], TransformConfig]:
    """Os controlos das transformações; devolve como criar a configuração."""
    numeric_column, categorical_column, scaling_column = st.columns(3)
    numeric = numeric_column.selectbox(
        "Nulos numéricos",
        [ImputationStrategy.MEDIAN, ImputationStrategy.MEAN, None],
        format_func=lambda option: "Não preencher" if option is None else option.label,
        key=f"{key}.numeric_imputation",
        persist_state="session",
    )
    categorical = categorical_column.selectbox(
        "Nulos categóricos",
        [ImputationStrategy.MOST_FREQUENT, None],
        format_func=lambda option: "Não preencher" if option is None else option.label,
        key=f"{key}.categorical_imputation",
        persist_state="session",
    )
    scaling = scaling_column.selectbox(
        "Escala",
        [ScalingMethod.STANDARD, ScalingMethod.MINMAX, None],
        format_func=lambda option: "Sem escala" if option is None else option.label,
        key=f"{key}.scaling",
        persist_state="session",
    )
    degree_column, dates_column = st.columns(2)
    degree = degree_column.slider(
        "Grau das features polinomiais",
        1,
        3,
        1,
        key=f"{key}.degree",
        persist_state="session",
        help="1 = sem features polinomiais; 2 junta x², x·y…",
    )
    date_features = dates_column.toggle(
        "Extrair ano, mês, dia e dia da semana das datas",
        value=True,
        key=f"{key}.date_features",
        persist_state="session",
    )
    numeric_features = [
        column.name
        for column in dataset.columns
        if column.kind is ColumnKind.NUMERIC and column.name not in (target, *dropped)
    ]
    binned = st.multiselect(
        "Dividir em intervalos (com os quantis do treino)",
        numeric_features,
        key=f"{key}.binned",
        persist_state="session",
    )
    n_bins = 5
    if binned:
        n_bins = st.slider(
            "Número de intervalos", 2, 20, 5, key=f"{key}.n_bins", persist_state="session"
        )
    return partial(
        TransformConfig,
        numeric_imputation=numeric,
        categorical_imputation=categorical,
        scaling=scaling,
        date_features=date_features,
        polynomial_degree=degree,
        binned_columns=tuple(binned),
        n_bins=n_bins,
    )


def _result(summary: PreparationSummary) -> None:
    columns = st.columns(len(summary.partitions) + 1)
    for column, part in zip(columns, summary.partitions, strict=False):
        column.metric(
            f"{part.partition.label} ({format_percent(part.ratio)})", format_count(part.n_rows)
        )
    columns[-1].metric("Features", format_count(len(summary.feature_names)))
    render_notices(summary.notices)

    lineage_tab, split_tab, data_tab, projection_tab = st.tabs(
        ["Linhagem", "Divisão", "Antes e depois", "Projeção 2D"]
    )
    with lineage_tab:
        st.dataframe(lineage_table(summary.lineage), hide_index=True, width="stretch")
        st.caption("Cada linha é uma operação, por ordem: é com isto que a preparação se repete.")
    with split_tab:
        _split_result(summary)
    with data_tab:
        st.markdown("**Depois da limpeza**")
        show_preview(summary.cleaned_preview, key="preparar.cleaned")
        st.markdown("**Treino depois das transformações (o que o modelo recebe)**")
        show_preview(summary.transformed_preview, key="preparar.transformed")
    with projection_tab:
        _projection(summary)
    st.page_link("pages/modelo.py", label="Escolher o modelo", icon=":material/arrow_forward:")


def _split_result(summary: PreparationSummary) -> None:
    if not summary.class_labels:
        st.markdown(
            ", ".join(
                f"{part.partition.label}: {format_count(part.n_rows)} linhas"
                for part in summary.partitions
            )
        )
        return
    table: dict[str, list[str]] = {"Classe": list(summary.class_labels)}
    for part in summary.partitions:
        table[part.partition.label] = [
            f"{format_percent(value.ratio)} ({format_count(value.count)})" for value in part.classes
        ]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Proporção de cada classe em cada parte. Na divisão estratificada, as percentagens "
        "são quase iguais em todas."
    )


def _projection(summary: PreparationSummary) -> None:
    points = summary.projection
    if points is None:
        st.info("Com menos de duas features, não há projeção 2D para mostrar.")
        return
    st.plotly_chart(projection_chart(points), width="stretch")
    st.caption(
        "Círculos: treino · losangos: validação · cruzes: teste. "
        + (
            "Os eixos são as duas componentes principais (PCA), aprendidas só com o treino."
            if points.method == "pca"
            else "Os eixos são as próprias features."
        )
    )
    if points.loadings is not None:
        st.markdown("**Peso de cada feature nas componentes**")
        st.dataframe(
            {
                "Feature": list(points.feature_names),
                points.axis_labels[0]: [format_number(w) for w in points.loadings[0]],
                points.axis_labels[1]: [format_number(w) for w in points.loadings[1]],
            },
            hide_index=True,
            width="stretch",
        )


main()
