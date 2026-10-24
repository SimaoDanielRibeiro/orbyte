"""Gráficos do ecrã Experiências: runs lado a lado (só desenham o que o ``RunComparison`` traz)."""

import plotly.graph_objects as go

from charts.distributions import CATEGORY_COLORS, styled
from ml_lab.core.services import RunComparison


def metrics_comparison_chart(comparison: RunComparison) -> go.Figure:
    """As métricas em percentagem, em barras agrupadas: uma cor por run."""
    rows = [metric for metric in comparison.metrics if metric.as_percent]
    figure = go.Figure()
    for index, label in enumerate(comparison.labels):
        figure.add_trace(
            go.Bar(
                name=label,
                x=[metric.display_name for metric in rows],
                y=[(metric.values[index] or 0.0) * 100 for metric in rows],
                marker_color=CATEGORY_COLORS[index % len(CATEGORY_COLORS)],
                hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>",
            )
        )
    figure.update_layout(
        barmode="group",
        yaxis_title="%",
        yaxis_range=[0, 100],
        legend={"orientation": "h", "y": 1.15},
    )
    return styled(figure)


def loss_comparison_chart(comparison: RunComparison) -> go.Figure:
    """A loss de cada run ao longo do treino (escala logarítmica se variar muito)."""
    figure = go.Figure()
    for index, series in enumerate(comparison.losses):
        figure.add_trace(
            go.Scatter(
                x=list(series.steps),
                y=list(series.losses),
                mode="lines",
                name=series.label,
                line={"color": CATEGORY_COLORS[index % len(CATEGORY_COLORS)]},
            )
        )
    values = [loss for series in comparison.losses for loss in series.losses if loss > 0]
    spread = max(values) / min(values) if values else 1.0
    figure.update_layout(
        xaxis_title="Passo",
        yaxis={"title": "Loss", "type": "log" if spread > 100 else "linear"},
        legend={"orientation": "h", "y": 1.15},
    )
    return styled(figure)
