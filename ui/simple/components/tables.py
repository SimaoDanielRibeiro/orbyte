"""Tabelas: pré-visualizações dos dados e linhagem, a partir das estruturas dos serviços."""

from collections.abc import Sequence
from typing import Any

import streamlit as st

from ml_lab.core.services import LineageStep, TablePreview
from ml_lab.formatting import format_count


def show_preview(preview: TablePreview, *, key: str | None = None) -> None:
    """Mostra as primeiras linhas de uma tabela e quantas linhas tem ao todo."""
    columns: dict[str, list[Any]] = {
        name: [row[index] for row in preview.rows] for index, name in enumerate(preview.columns)
    }
    st.dataframe(columns, hide_index=True, width="stretch", key=key)
    st.caption(
        f"Primeiras {format_count(len(preview.rows))} de {format_count(preview.n_rows)} linhas."
    )


def lineage_table(steps: Sequence[LineageStep]) -> dict[str, list[str]]:
    """A linhagem como tabela: etapa, o que aconteceu, linhas e colunas."""
    table: dict[str, list[str]] = {
        "Etapa": [],
        "O que aconteceu": [],
        "Linhas": [],
        "Colunas novas": [],
        "Colunas removidas": [],
    }
    for step in steps:
        rows = ""
        if step.rows_before is not None and step.rows_after is not None:
            if step.rows_before == step.rows_after:
                rows = format_count(step.rows_after)
            else:
                rows = f"{format_count(step.rows_before)} → {format_count(step.rows_after)}"
        elif step.rows_after is not None:
            rows = format_count(step.rows_after)
        table["Etapa"].append(step.stage.label)
        table["O que aconteceu"].append(step.description)
        table["Linhas"].append(rows)
        table["Colunas novas"].append(", ".join(step.columns_added))
        table["Colunas removidas"].append(", ".join(step.columns_removed))
    return table
