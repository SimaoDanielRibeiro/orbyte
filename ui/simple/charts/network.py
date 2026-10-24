"""Gráficos das camadas de uma rede neuronal (ecrã Treinar): só desenham o ``NetworkTrace``."""

from collections.abc import Sequence
from itertools import pairwise

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charts.distributions import CATEGORY_COLORS, styled
from ml_lab.core.services import NetworkTrace
from ml_lab.formatting import format_percent

# Neurónios desenhados por camada, no máximo (com mais, o diagrama deixa de se ler).
_MAX_DRAWN = 24
_STUCK = "#9E9E9E"
_NEUTRAL = "#BAB0AC"


def network_chart(trace: NetworkTrace, index: int) -> go.Figure:
    """O diagrama da rede na fotografia ``index``: neurónios pintados com a sua ativação média."""
    frame = trace.frames[index]
    sizes = [trace.inputs, *(layer.units for layer in trace.layers)]
    positions = [_drawn(size) for size in sizes]
    figure = go.Figure()
    edges_x: list[float | None] = []
    edges_y: list[float | None] = []
    for column, (left, right) in enumerate(pairwise(positions)):
        for _, y_left in left:
            for _, y_right in right:
                edges_x += [column, column + 1, None]
                edges_y += [y_left, y_right, None]
    figure.add_trace(
        go.Scatter(
            x=edges_x,
            y=edges_y,
            mode="lines",
            line={"color": "rgba(150, 150, 150, 0.15)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(_column(0, positions[0], "Entrada", _NEUTRAL))
    lowest, highest = _activation_range(trace)
    for column, (layer, layer_frame) in enumerate(
        zip(trace.layers, frame.layers, strict=True), start=1
    ):
        units = positions[column]
        if not layer.hidden:
            figure.add_trace(_column(column, units, f"{layer.name} ({layer.activation})", _NEUTRAL))
            continue
        values = [layer_frame.activations[unit] for unit, _ in units]
        stuck = [
            unit < len(layer_frame.stuck_units) and layer_frame.stuck_units[unit]
            for unit, _ in units
        ]
        figure.add_trace(
            go.Scatter(
                x=[column] * len(units),
                y=[y for _, y in units],
                mode="markers",
                marker={
                    "size": 16,
                    "color": values,
                    "colorscale": "Viridis",
                    "cmin": lowest,
                    "cmax": highest,
                    "showscale": column == 1,
                    "colorbar": {"title": {"text": "Ativação média", "side": "right"}},
                    "line": {
                        "color": [_STUCK if flag else "#888888" for flag in stuck],
                        "width": 2,
                    },
                    "opacity": [0.35 if flag else 1.0 for flag in stuck],
                },
                customdata=[
                    [unit + 1, "parado" if flag else ""]
                    for (unit, _), flag in zip(units, stuck, strict=True)
                ],
                hovertemplate=(
                    f"{layer.name}, neurónio %{{customdata[0]}}: ativação média "
                    "%{marker.color:.3f} %{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    names = ["Entrada", *(_header(layer.name, layer.units) for layer in trace.layers)]
    figure.update_layout(
        xaxis={"tickvals": list(range(len(names))), "ticktext": names, "showgrid": False},
        yaxis={"visible": False},
    )
    title = f"Passo {frame.step} · {format_percent(frame.stuck)} dos neurónios escondidos parados"
    return styled(figure, height=420, title=title)


def gradients_chart(trace: NetworkTrace) -> go.Figure:
    """O tamanho do gradiente dos pesos de cada camada, época a época (escala logarítmica)."""
    steps = [frame.step for frame in trace.frames]
    figure = go.Figure()
    for position, layer in enumerate(trace.layers):
        values = [frame.layers[position].gradient for frame in trace.frames]
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines+markers",
                name=layer.name,
                line={"color": CATEGORY_COLORS[position % len(CATEGORY_COLORS)]},
            )
        )
    figure.update_layout(
        xaxis_title="Passo do treino",
        yaxis={"title": "Norma do gradiente", "type": "log", "exponentformat": "power"},
        legend={"orientation": "h", "y": 1.15},
    )
    return styled(figure)


def stuck_chart(trace: NetworkTrace) -> go.Figure:
    """A percentagem de neurónios parados em cada camada escondida, época a época."""
    steps = [frame.step for frame in trace.frames]
    figure = go.Figure()
    for position, layer in enumerate(trace.layers):
        if not layer.hidden:
            continue
        values = [frame.layers[position].stuck for frame in trace.frames]
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=[None if value is None else value * 100 for value in values],
                mode="lines+markers",
                name=layer.name,
                line={"color": CATEGORY_COLORS[position % len(CATEGORY_COLORS)]},
            )
        )
    figure.update_layout(
        xaxis_title="Passo do treino",
        yaxis={"title": "% de neurónios parados", "range": [0, 100]},
        legend={"orientation": "h", "y": 1.15},
    )
    return styled(figure)


def activations_chart(trace: NetworkTrace, index: int) -> go.Figure:
    """A distribuição das ativações de cada camada escondida, na fotografia ``index``."""
    hidden = [position for position, layer in enumerate(trace.layers) if layer.hidden]
    figure = make_subplots(
        rows=1, cols=len(hidden), subplot_titles=[trace.layers[p].name for p in hidden]
    )
    frame = trace.frames[index]
    for column, position in enumerate(hidden, start=1):
        layer = frame.layers[position]
        centers = [(left + right) / 2 for left, right in pairwise(layer.edges)]
        figure.add_trace(
            go.Bar(
                x=centers,
                y=list(layer.counts),
                marker_color=CATEGORY_COLORS[position % len(CATEGORY_COLORS)],
                showlegend=False,
                hovertemplate="ativação %{x:.2f}: %{y}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(title_text="Ativação")
    figure.update_yaxes(title_text="Quantas", col=1)
    styled(figure, height=300)
    figure.update_layout(margin_t=40)  # o nome de cada camada fica por cima do seu gráfico
    return figure


def _drawn(units: int) -> list[tuple[int, float]]:
    """Os neurónios desenhados de uma camada (no máximo ``_MAX_DRAWN``) e a altura de cada um."""
    if units <= _MAX_DRAWN:
        chosen = list(range(units))
    else:
        step = units / _MAX_DRAWN
        chosen = [int(i * step) for i in range(_MAX_DRAWN)]
    middle = (len(chosen) - 1) / 2
    return [(unit, middle - row) for row, unit in enumerate(chosen)]


def _column(column: int, units: Sequence[tuple[int, float]], name: str, color: str) -> go.Scatter:
    return go.Scatter(
        x=[column] * len(units),
        y=[y for _, y in units],
        mode="markers",
        marker={"size": 14, "color": color, "line": {"color": "#888888", "width": 1}},
        hovertemplate=f"{name}<extra></extra>",
        showlegend=False,
    )


def _activation_range(trace: NetworkTrace) -> tuple[float, float]:
    """A mesma escala de cores em todas as fotografias, para se poderem comparar."""
    values = [
        value for frame in trace.frames for layer in frame.layers for value in layer.activations
    ]
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def _header(name: str, units: int) -> str:
    return f"{name}<br>{units} neurónios"
