"""Gráficos do perfil: distribuições, correlações, nulos e classes.

Só desenham: os números (contagens, intervalos, correlações) chegam já calculados
pelos serviços, para a interface aprimorada poder desenhar os mesmos.
"""

from collections.abc import Sequence

import plotly.graph_objects as go

from ml_lab.core.services import (
    ClassBalance,
    ColumnProfile,
    CorrelationMatrix,
    Histogram,
    MissingMap,
    ValueCount,
)

PRIMARY = "#4C78A8"
MISSING = "#E45756"

CATEGORY_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)
"""Cores bem distintas para classes (paleta Tableau 10), iguais em todos os gráficos."""


def styled(figure: go.Figure, *, height: int = 320, title: str | None = None) -> go.Figure:
    """Aspeto comum: margens pequenas e números à portuguesa (vírgula decimal)."""
    figure.update_layout(
        height=height,
        title=title,
        margin={"l": 10, "r": 10, "t": 40 if title else 10, "b": 10},
        separators=", ",
    )
    return figure


def histogram_chart(histogram: Histogram, column: str) -> go.Figure:
    """Histograma: quantas linhas caem em cada intervalo."""
    edges = histogram.edges
    lows, highs = edges[:-1], edges[1:]
    figure = go.Figure(
        go.Bar(
            x=[(low + high) / 2 for low, high in zip(lows, highs, strict=True)],
            y=list(histogram.counts),
            width=[high - low for low, high in zip(lows, highs, strict=True)],
            customdata=[[low, high] for low, high in zip(lows, highs, strict=True)],
            marker_color=PRIMARY,
            hovertemplate="de %{customdata[0]:.3~f} a %{customdata[1]:.3~f}<br>%{y} linhas"
            "<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title=column, yaxis_title="Linhas", bargap=0.02)
    return styled(figure)


def value_counts_chart(values: Sequence[ValueCount], column: str) -> go.Figure:
    """Barras com os valores mais frequentes."""
    ordered = list(reversed(values))
    figure = go.Figure(
        go.Bar(
            x=[value.count for value in ordered],
            y=[value.value for value in ordered],
            orientation="h",
            customdata=[value.ratio * 100 for value in ordered],
            marker_color=PRIMARY,
            hovertemplate="%{y}: %{x} linhas (%{customdata:.1f}%)<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="Linhas", yaxis_title=column, yaxis_type="category")
    return styled(figure, height=max(220, 32 * len(ordered) + 80))


def correlation_chart(matrix: CorrelationMatrix) -> go.Figure:
    """Mapa de correlações: vermelho é positiva, azul é negativa."""
    labels = list(matrix.columns)
    figure = go.Figure(
        go.Heatmap(
            z=[list(row) for row in matrix.values],
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            texttemplate="%{z:.2f}",
            hovertemplate="%{y} e %{x}: r = %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(yaxis_autorange="reversed")
    return styled(figure, height=max(320, 40 * len(labels) + 120))


def missing_ratio_chart(columns: Sequence[ColumnProfile]) -> go.Figure:
    """Percentagem de valores em falta por coluna."""
    figure = go.Figure(
        go.Bar(
            x=[column.name for column in columns],
            y=[column.missing_ratio * 100 for column in columns],
            customdata=[column.n_missing for column in columns],
            marker_color=MISSING,
            hovertemplate="%{x}: %{y:.1f}% em falta (%{customdata} linhas)<extra></extra>",
        )
    )
    figure.update_layout(yaxis_title="% em falta", yaxis_range=[0, 100])
    return styled(figure)


def missing_map_chart(missing: MissingMap) -> go.Figure:
    """Mapa de nulos: cada linha do gráfico é uma linha dos dados; vermelho é um valor em falta."""
    figure = go.Figure(
        go.Heatmap(
            z=[[1 if value else 0 for value in row] for row in missing.missing],
            x=list(missing.columns),
            y=list(missing.rows),
            zmin=0,
            zmax=1,
            colorscale=[[0, "#F1F3F5"], [1, MISSING]],
            showscale=False,
            hovertemplate="linha %{y}, %{x}<extra></extra>",
        )
    )
    figure.update_layout(yaxis_title="Linha", yaxis_autorange="reversed")
    return styled(figure, height=420)


def class_balance_chart(balance: ClassBalance) -> go.Figure:
    """Linhas por classe do alvo."""
    figure = go.Figure(
        go.Bar(
            x=[value.value for value in balance.classes],
            y=[value.count for value in balance.classes],
            customdata=[value.ratio * 100 for value in balance.classes],
            marker_color=PRIMARY,
            hovertemplate="%{x}: %{y} linhas (%{customdata:.1f}%)<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title=balance.target, yaxis_title="Linhas", xaxis_type="category")
    return styled(figure)
