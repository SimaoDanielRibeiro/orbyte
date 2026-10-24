"""Gráficos do treino: a fronteira (ou a previsão) em cada snapshot e a curva da loss.

A animação corre no navegador: cada snapshot é um *frame* do Plotly, com um slider
para percorrer o treino e um botão para o reproduzir. Os números chegam prontos de
``core/visualization_data`` (o ``TrainingAnimation``); aqui só se desenha.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

from charts.distributions import CATEGORY_COLORS, PRIMARY, styled
from ml_lab.core.services import FieldFrame, StepRecord, TrainingAnimation
from ml_lab.formatting import format_count, format_significant

# Milissegundos por fotografia na reprodução; a cor da fronteira de decisão no plano (por
# cima do branco do mapa, onde a probabilidade é 50%) e a das curvas, que tem de se ver
# no tema claro e no escuro.
_FRAME_MS = 150
_BOUNDARY = "#222222"
_CURVE = CATEGORY_COLORS[3]


def animation_chart(animation: TrainingAnimation) -> go.Figure:
    """O campo do modelo por cima dos pontos de treino; com vários snapshots, animado."""
    figure = _plane(animation) if animation.dimensions == 2 else _line(animation)
    figure.update_layout(showlegend=bool(animation.points.labels))
    if animation.animated:
        frames = []
        for frame in animation.frames:
            data = _field(animation, frame)
            frames.append(
                go.Frame(
                    data=data,
                    traces=list(range(len(data))),
                    name=str(frame.step),
                    layout={"title": {"text": _title(frame)}},
                )
            )
        figure.frames = frames
        _add_player(figure, animation.frames)
    return styled(figure, height=520, title=_title(animation.frames[-1]))


def loss_chart(
    history: Sequence[StepRecord], *, percent_metrics: Mapping[str, str] | None = None
) -> go.Figure:
    """A loss em cada passo e, à direita, as métricas parciais em percentagem.

    Args:
        history: Os passos do treino.
        percent_metrics: Que métricas parciais desenhar (nome → nome a mostrar).
    """
    steps = [record.step for record in history]
    figure = go.Figure(
        go.Scatter(
            x=steps,
            y=[record.loss for record in history],
            mode="lines",
            name="Loss (treino)",
            line={"color": PRIMARY, "width": 2},
            hovertemplate="passo %{x}: loss %{y:.4~g}<extra></extra>",
        )
    )
    for number, (name, label) in enumerate((percent_metrics or {}).items(), start=1):
        values = [record.metrics.get(name) for record in history]
        if all(value is None for value in values):
            continue
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=[None if value is None else value * 100 for value in values],
                mode="lines",
                name=f"{label} (treino)",
                yaxis="y2",
                line={"color": CATEGORY_COLORS[number], "width": 1.5, "dash": "dot"},
                hovertemplate=f"passo %{{x}}: {label.lower()} %{{y:.1f}}%<extra></extra>",
            )
        )
    figure.update_layout(
        xaxis_title="Passo do treino",
        # Potências de 10 (4×10⁹), e não as abreviaturas inglesas (4B).
        yaxis={"title": "Loss", "exponentformat": "power"},
        yaxis2={
            "title": "%",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "tickmode": "linear",
            "tick0": 0,
            "dtick": 25,
            "showgrid": False,
        },
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=320)


def _plane(animation: TrainingAnimation) -> go.Figure:
    figure = go.Figure(data=_field(animation, animation.frames[-1]))
    points = animation.points
    if points.labels:
        for number, label in enumerate(animation.class_labels):
            chosen = [i for i, value in enumerate(points.labels) if value == label]
            if not chosen:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[points.x[i] for i in chosen],
                    y=[points.y[i] for i in chosen],
                    mode="markers",
                    name=label,
                    marker={
                        "size": 7,
                        "color": CATEGORY_COLORS[number % len(CATEGORY_COLORS)],
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate=f"classe {label}<extra></extra>",
                )
            )
    else:
        low, high = _value_range(animation)
        figure.add_trace(
            go.Scatter(
                x=list(points.x),
                y=list(points.y),
                mode="markers",
                name="Treino",
                marker={
                    "size": 7,
                    "color": list(points.values),
                    "colorscale": "Viridis",
                    "cmin": low,
                    "cmax": high,
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate="valor real %{marker.color:.3~f}<extra></extra>",
            )
        )
    x_title, y_title = animation.axis_labels
    figure.update_layout(
        xaxis={"title": x_title, "range": [animation.xs[0], animation.xs[-1]]},
        yaxis={"title": y_title, "range": [animation.ys[0], animation.ys[-1]]},
        # Por cima do gráfico, à direita: a barra de cores ocupa a lateral.
        legend={
            "title": "Classe",
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.02,
            "yanchor": "bottom",
        },
    )
    return figure


def _line(animation: TrainingAnimation) -> go.Figure:
    figure = go.Figure(data=_field(animation, animation.frames[-1]))
    points = animation.points
    if points.labels:
        codes = {label: number for number, label in enumerate(animation.class_labels)}
        for label, number in codes.items():
            chosen = [i for i, value in enumerate(points.labels) if value == label]
            if not chosen:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[points.x[i] for i in chosen],
                    y=[float(number)] * len(chosen),
                    mode="markers",
                    name=label,
                    marker={"size": 8, "color": CATEGORY_COLORS[number % len(CATEGORY_COLORS)]},
                    hovertemplate=f"classe {label}: %{{x:.3~f}}<extra></extra>",
                )
            )
    else:
        figure.add_trace(
            go.Scatter(
                x=list(points.x),
                y=list(points.y),
                mode="markers",
                name="Treino",
                marker={"size": 6, "color": CATEGORY_COLORS[0], "opacity": 0.6},
                hovertemplate="%{x:.3~f} → %{y:.3~f}<extra></extra>",
            )
        )
    x_title, y_title = animation.axis_labels
    yaxis: dict[str, Any] = {"title": y_title}
    if animation.value_kind == "probability":
        yaxis |= {"title": animation.value_label, "range": [-0.05, 1.05]}
        figure.add_hline(y=0.5, line={"dash": "dash", "color": "#888888", "width": 1})
    elif animation.value_kind == "class":
        yaxis |= {
            "title": animation.value_label,
            "tickvals": list(range(len(animation.class_labels))),
            "ticktext": list(animation.class_labels),
        }
    else:
        yaxis["range"] = list(_value_range(animation))
    figure.update_layout(
        xaxis={"title": x_title, "range": [animation.xs[0], animation.xs[-1]]}, yaxis=yaxis
    )
    return figure


def _field(animation: TrainingAnimation, frame: FieldFrame) -> list[BaseTraceType]:
    """O que muda de fotografia para fotografia (os primeiros traços da figura)."""
    xs = list(animation.xs)
    if animation.dimensions == 1:
        values = list(frame.values[0])
        return [
            go.Scatter(
                x=xs,
                y=values,
                mode="lines",
                name=animation.value_label,
                line={
                    "color": _CURVE,
                    "width": 3,
                    "shape": "hv" if animation.value_kind == "class" else "linear",
                },
                showlegend=False,
                hovertemplate="%{x:.3~f} → %{y:.3~f}<extra></extra>",
            )
        ]
    z = [list(row) for row in frame.values]
    ys = list(animation.ys)
    if animation.value_kind == "probability":
        first, second = CATEGORY_COLORS[0], CATEGORY_COLORS[1]
        return [
            go.Heatmap(
                x=xs,
                y=ys,
                z=z,
                zmin=0,
                zmax=1,
                zsmooth="best",
                colorscale=[[0, first], [0.5, "#ffffff"], [1, second]],
                opacity=0.7,
                colorbar={"title": {"text": animation.value_label, "side": "right"}},
                hovertemplate="probabilidade %{z:.2f}<extra></extra>",
            ),
            go.Contour(
                x=xs,
                y=ys,
                z=z,
                contours={"start": 0.5, "end": 0.5, "size": 1, "coloring": "lines"},
                colorscale=[[0, _BOUNDARY], [1, _BOUNDARY]],
                line={"width": 2},
                showscale=False,
                hoverinfo="skip",
            ),
        ]
    if animation.value_kind == "class":
        n_classes = max(1, len(animation.class_labels))
        bands: list[list[float | str]] = []
        for number in range(n_classes):
            color = CATEGORY_COLORS[number % len(CATEGORY_COLORS)]
            bands += [[number / n_classes, color], [(number + 1) / n_classes, color]]
        return [
            go.Heatmap(
                x=xs,
                y=ys,
                z=z,
                zmin=-0.5,
                zmax=n_classes - 0.5,
                colorscale=bands,
                opacity=0.35,
                showscale=False,
                hoverinfo="skip",
            )
        ]
    low, high = _value_range(animation)
    return [
        go.Heatmap(
            x=xs,
            y=ys,
            z=z,
            zmin=low,
            zmax=high,
            zsmooth="best",
            colorscale="Viridis",
            opacity=0.8,
            colorbar={"title": {"text": animation.value_label, "side": "right"}},
            hovertemplate="previsão %{z:.3~f}<extra></extra>",
        )
    ]


def _value_range(animation: TrainingAnimation) -> tuple[float, float]:
    """A escala dos valores: a dos dados, alargada até uma vez a sua extensão para cada lado.

    Os campos fora disto (um treino que explodiu) ficam cortados, em vez de esmagarem a
    escala e esconderem os dados.
    """
    data = list(animation.points.values) or list(animation.points.y)
    low, high = min(data, default=0.0), max(data, default=1.0)
    span = (high - low) or 1.0
    values = [v for frame in animation.frames for row in frame.values for v in row]
    low = max(min([low, *values]), low - span)
    high = min(max([high, *values]), high + span)
    if animation.dimensions == 1:
        margin = (high - low) * 0.05
        return low - margin, high + margin
    return low, high


def _title(frame: FieldFrame) -> str:
    if frame.step == 0:
        return "Modelo final"
    loss = "" if frame.loss is None else f" · loss {format_significant(frame.loss)}"
    return f"Passo {format_count(frame.step)}{loss}"


def _add_player(figure: go.Figure, frames: Sequence[FieldFrame]) -> None:
    """Slider para percorrer as fotografias e botões para as reproduzir."""
    still = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
    figure.update_layout(
        sliders=[
            {
                "active": len(frames) - 1,
                "currentvalue": {"prefix": "Passo "},
                "pad": {"t": 40},
                "steps": [
                    {
                        "label": format_count(frame.step),
                        "method": "animate",
                        "args": [[str(frame.step)], still],
                    }
                    for frame in frames
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": 0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 40, "r": 10},
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Reproduzir",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": _FRAME_MS, "redraw": True},
                                "fromcurrent": False,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pausa",
                        "method": "animate",
                        "args": [[None], still],
                    },
                ],
            }
        ],
    )
