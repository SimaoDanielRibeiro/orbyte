"""Gráfico da projeção 2D: cada ponto é uma linha dos dados, colorida pela classe ou pelo alvo."""

import plotly.graph_objects as go

from charts.distributions import CATEGORY_COLORS, styled
from ml_lab.core.services import Partition, ProjectionPoints
from ml_lab.formatting import format_percent

# Símbolo de cada parte: o treino em círculos, a validação em losangos, o teste em cruzes.
_SYMBOLS = {Partition.TRAIN: "circle", Partition.VALIDATION: "diamond", Partition.TEST: "x"}


def projection_chart(points: ProjectionPoints) -> go.Figure:
    """Dispersão 2D dos dados preparados."""
    symbols = [_SYMBOLS[part] for part in points.partitions]
    parts = [part.label for part in points.partitions]
    figure = go.Figure()
    if points.labels:
        for number, label in enumerate(sorted(set(points.labels))):
            chosen = [i for i, value in enumerate(points.labels) if value == label]
            color = CATEGORY_COLORS[number % len(CATEGORY_COLORS)]
            figure.add_trace(
                go.Scattergl(
                    x=[points.x[i] for i in chosen],
                    y=[points.y[i] for i in chosen],
                    mode="markers",
                    name=label,
                    marker={
                        "symbol": [symbols[i] for i in chosen],
                        "size": 7,
                        "opacity": 0.8,
                        "color": color,
                    },
                    customdata=[parts[i] for i in chosen],
                    hovertemplate=f"classe {label} (%{{customdata}})<extra></extra>",
                )
            )
    else:
        marker: dict[str, object] = {"symbol": symbols, "size": 7, "opacity": 0.8}
        if points.values:
            marker |= {"color": list(points.values), "colorscale": "Viridis", "showscale": True}
        figure.add_trace(
            go.Scattergl(
                x=list(points.x),
                y=list(points.y),
                mode="markers",
                marker=marker,
                customdata=parts,
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )
    x_title, y_title = points.axis_labels
    if points.explained_variance_ratio is not None:
        first, second = points.explained_variance_ratio
        x_title = f"{x_title} ({format_percent(first)} da variância)"
        y_title = f"{y_title} ({format_percent(second)} da variância)"
    figure.update_layout(xaxis_title=x_title, yaxis_title=y_title, legend_title="Classe")
    return styled(figure, height=460)
