"""Ecrã 3 · Explorar: perfil das colunas, distribuições e avisos."""

import streamlit as st

from charts.distributions import (
    class_balance_chart,
    correlation_chart,
    histogram_chart,
    missing_map_chart,
    missing_ratio_chart,
    value_counts_chart,
)
from components.layout import render_header, render_why, require_dataset
from components.notices import render_notices
from components.state import current_preparation, get_lab
from ml_lab.core.services import ColumnKind, ColumnProfile, DatasetProfile, DatasetSummary
from ml_lab.formatting import format_count, format_number, format_percent, format_quantity

# Singular e plural de cada tipo, para frases como "5 numéricas, 1 booleana".
_KIND_WORDS = {
    ColumnKind.NUMERIC: ("numérica", "numéricas"),
    ColumnKind.CATEGORICAL: ("categórica", "categóricas"),
    ColumnKind.BOOLEAN: ("booleana", "booleanas"),
    ColumnKind.DATETIME: ("de datas", "de datas"),
    ColumnKind.TEXT: ("de texto", "de texto"),
    ColumnKind.IDENTIFIER: ("identificador", "identificadores"),
}


def main() -> None:
    """Desenha o ecrã."""
    render_header("explorar")
    dataset = require_dataset()
    if dataset is None:
        return
    render_why("profiling")
    target = _target(dataset)
    profile = get_lab().datasets.profile(dataset.id, target)
    _overview(profile)
    st.subheader("Avisos")
    notices = profile.all_notices
    if notices:
        render_notices(notices)
    else:
        st.success(
            "O perfil não encontrou nada que mereça atenção.", icon=":material/check_circle:"
        )

    columns_tab, correlations_tab, missing_tab, target_tab = st.tabs(
        ["Colunas", "Correlações", "Valores em falta", "Alvo"]
    )
    with columns_tab:
        _columns(profile, dataset.id)
    with correlations_tab:
        if profile.correlations is None:
            st.info("São precisas pelo menos duas colunas numéricas para medir correlações.")
        else:
            st.plotly_chart(correlation_chart(profile.correlations), width="stretch")
            st.caption(
                "Correlação de Pearson: 1 é uma relação linear perfeita, -1 perfeita mas ao "
                "contrário, 0 é nenhuma relação linear. Correlação não é causalidade."
            )
    with missing_tab:
        _missing(profile)
    with target_tab:
        _target_profile(profile)


def _target(dataset: DatasetSummary) -> str | None:
    options: list[str | None] = [None, *(column.name for column in dataset.columns)]
    preparation = current_preparation()
    preferred = preparation.config.target if preparation else dataset.suggested_target
    return st.selectbox(
        "Coluna-alvo (para ver o equilíbrio das classes)",
        options,
        index=options.index(preferred) if preferred in options else 0,
        format_func=lambda option: "Sem alvo" if option is None else option,
        key=f"explorar.{dataset.id}.target",
        persist_state="session",
    )


def _overview(profile: DatasetProfile) -> None:
    rows, columns, missing, duplicates, memory = st.columns(5)
    rows.metric("Linhas", format_count(profile.n_rows))
    columns.metric("Colunas", format_count(profile.n_columns))
    missing.metric("Células em falta", format_percent(profile.missing_ratio))
    duplicates.metric("Linhas duplicadas", format_count(profile.n_duplicate_rows))
    memory.metric("Memória", f"{format_number(profile.memory_bytes / 1024**2, 1)} MB")
    counts = sorted(profile.kind_counts.items(), key=lambda item: -item[1])
    kinds = ", ".join(format_quantity(count, *_KIND_WORDS[kind]) for kind, count in counts)
    st.caption(f"Colunas por tipo inferido: {kinds}.")


def _columns(profile: DatasetProfile, dataset_id: str) -> None:
    st.dataframe(
        {
            "Coluna": [c.name for c in profile.columns],
            "Tipo": [c.kind.label for c in profile.columns],
            "Em falta": [format_percent(c.missing_ratio) for c in profile.columns],
            "Distintos": [c.n_unique for c in profile.columns],
            "Tipo no pandas": [c.dtype for c in profile.columns],
        },
        hide_index=True,
        width="stretch",
    )
    names = [column.name for column in profile.columns]
    name = st.selectbox(
        "Ver uma coluna em pormenor",
        names,
        key=f"explorar.{dataset_id}.column",
        persist_state="session",
    )
    _column_detail(profile.column(name))


def _column_detail(column: ColumnProfile) -> None:
    st.markdown(f"**{column.name}** · {column.kind.label}: {column.kind_reason}")
    summary = column.numeric
    if summary is not None:
        stats = {
            "Média": summary.mean,
            "Desvio padrão": summary.std,
            "Mínimo": summary.minimum,
            "1.º quartil": summary.q1,
            "Mediana": summary.median,
            "3.º quartil": summary.q3,
            "Máximo": summary.maximum,
        }
        if summary.skewness is not None:
            stats["Assimetria"] = summary.skewness
        st.dataframe(
            {"Estatística": list(stats), "Valor": [format_number(v) for v in stats.values()]},
            hide_index=True,
        )
    if column.kind is ColumnKind.IDENTIFIER:
        st.caption("Um identificador tem um valor diferente em cada linha: não há distribuição.")
    elif column.histogram is not None:
        st.plotly_chart(histogram_chart(column.histogram, column.name), width="stretch")
    elif column.top_values:
        st.plotly_chart(value_counts_chart(column.top_values, column.name), width="stretch")
        if column.n_unique > len(column.top_values):
            st.caption(f"Os {len(column.top_values)} valores mais frequentes de {column.n_unique}.")
    if column.date_range is not None:
        first, last = column.date_range
        st.markdown(f"De **{first}** a **{last}**.")
    render_notices(column.notices)


def _missing(profile: DatasetProfile) -> None:
    if not profile.n_missing_cells:
        st.success("Não há valores em falta.", icon=":material/check_circle:")
        return
    st.plotly_chart(missing_ratio_chart(profile.columns), width="stretch")
    st.markdown("**Onde estão os nulos**")
    st.plotly_chart(missing_map_chart(profile.missing_map), width="stretch")
    st.caption(
        f"{len(profile.missing_map.rows)} linhas espalhadas pelo dataset. Faixas vermelhas na "
        "vertical são colunas com muitos nulos; na horizontal, linhas quase vazias."
    )


def _target_profile(profile: DatasetProfile) -> None:
    if profile.target is None:
        st.info("Escolhe uma coluna-alvo no topo do ecrã para ver as classes.")
        return
    balance = profile.class_balance
    if balance is None:
        st.markdown(
            f"**{profile.target}** é numérica contínua (regressão): não há classes para contar."
        )
        _column_detail(profile.column(profile.target))
        return
    st.plotly_chart(class_balance_chart(balance), width="stretch")
    st.caption(
        f"Equilíbrio: a classe menos frequente tem {format_percent(balance.imbalance_ratio)} das "
        "linhas da mais frequente (100% é perfeitamente equilibrado)."
    )
    means = profile.group_means
    if means is not None:
        st.markdown(f"**Média de cada feature numérica por classe de {profile.target}**")
        table: dict[str, list[str]] = {"Classe": list(means.groups)}
        for index, column in enumerate(means.columns):
            table[column] = [
                "—" if (value := row[index]) is None else format_number(value)
                for row in means.means
            ]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Diferenças grandes entre classes indicam features que ajudam a separá-las.")


main()
