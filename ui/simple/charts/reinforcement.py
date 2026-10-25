"""Gráficos do reinforcement learning (ecrã Reinforcement learning): só desenham o que o serviço dá.

A grelha desenha-se com a linha 0 em cima, como num mapa. A cor de cada célula é o seu valor
(o maior Q), sempre na mesma escala, de −1 a 1, para os episódios se poderem comparar.
"""

from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charts.distributions import CATEGORY_COLORS, styled
from ml_lab.core.services import CellKind, GridLayout, PolicyFrame, RewardCurve, StepRecord

_VALUES = [[0.0, "#E45756"], [0.5, "#F7F7F7"], [1.0, "#4C78A8"]]
"""Vermelho para o que se espera perder, azul para o que se espera ganhar."""

_CELLS = {CellKind.WALL: "#5B5B5B", CellKind.GOAL: "#54A24B", CellKind.TRAP: "#E45756"}
_ARROWS = {(-1, 0): "↑", (0, 1): "→", (1, 0): "↓", (0, -1): "←"}
_PATH = "#222222"


def reward_chart(curve: RewardCurve) -> go.Figure:
    """A recompensa de cada episódio, a média móvel e, à direita, quantos chegaram ao objetivo."""
    episodes = list(curve.episodes)
    figure = go.Figure(
        [
            go.Scatter(
                x=episodes,
                y=list(curve.rewards),
                mode="markers",
                name="Recompensa do episódio",
                marker={"size": 4, "color": CATEGORY_COLORS[0], "opacity": 0.35},
                hovertemplate="episódio %{x}: %{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=episodes,
                y=list(curve.average),
                mode="lines",
                name=f"Média ({curve.window} episódios)",
                line={"color": CATEGORY_COLORS[0], "width": 3},
                hovertemplate="episódio %{x}: média %{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=episodes,
                y=[value * 100 for value in curve.success],
                mode="lines",
                name="Chegou ao objetivo",
                yaxis="y2",
                line={"color": CATEGORY_COLORS[2], "width": 1.5, "dash": "dot"},
                hovertemplate="episódio %{x}: %{y:.0f}% chegaram<extra></extra>",
            ),
        ]
    )
    figure.update_layout(
        xaxis_title="Episódio",
        yaxis={"title": "Recompensa"},
        yaxis2={
            "title": "% que chegou",
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
            # Marcas próprias, de 25 em 25: sem isto, o Plotly alinha-as com as da esquerda.
            "tickmode": "linear",
            "tick0": 0,
            "dtick": 25,
            "showgrid": False,
        },
        legend={"orientation": "h", "y": 1.15},
    )
    return styled(figure, height=340)


def live_reward_chart(steps: Sequence[StepRecord]) -> go.Figure:
    """A recompensa dos episódios que já correram (durante o treino)."""
    figure = go.Figure(
        go.Scatter(
            x=[record.step for record in steps],
            y=[record.metrics.get("reward") for record in steps],
            mode="markers",
            marker={"size": 4, "color": CATEGORY_COLORS[0], "opacity": 0.5},
            hovertemplate="episódio %{x}: %{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="Episódio", yaxis_title="Recompensa")
    return styled(figure, height=300)


def policy_chart(layout: GridLayout, frame: PolicyFrame) -> go.Figure:
    """A grelha: o valor de cada célula, a política (as setas) e o caminho a partir da partida."""
    figure = go.Figure(
        go.Heatmap(
            z=_grid(layout, frame.values),
            zmin=-1,
            zmax=1,
            colorscale=_VALUES,
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": "Valor", "side": "right"}},
            hovertemplate="linha %{y}, coluna %{x}: valor %{z:.3f}<extra></extra>",
        )
    )
    for state, kind in enumerate(layout.cells):
        if kind in _CELLS:  # sem valor (não há nada a escolher): a cor diz o que é
            row, column = layout.position(state)
            figure.add_shape(
                type="rect",
                x0=column - 0.5,
                x1=column + 0.5,
                y0=row - 0.5,
                y1=row + 0.5,
                fillcolor=_CELLS[kind],
                line={"width": 0},
                layer="below",
            )
    arrows = [
        (*layout.position(state), _ARROWS.get(layout.action_moves[action], "?"))
        for state, action in enumerate(frame.actions)
        if action is not None
    ]
    figure.add_trace(
        go.Scatter(
            x=[column for _, column, _ in arrows],
            y=[row for row, _, _ in arrows],
            mode="text",
            text=[arrow for _, _, arrow in arrows],
            textfont={"size": 18, "color": "#333333"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    path = [layout.position(state) for state in frame.path]
    figure.add_trace(
        go.Scatter(
            x=[column for _, column in path],
            y=[row for row, _ in path],
            mode="lines+markers",
            line={"color": _PATH, "width": 3},
            marker={"size": 7, "color": _PATH},
            name="Caminho aprendido",
            hoverinfo="skip",
        )
    )
    start = layout.position(layout.cells.index(CellKind.START))
    figure.add_annotation(x=start[1], y=start[0], text="partida", showarrow=False, yshift=-20)
    _grid_axes(figure, layout)
    figure.update_layout(showlegend=False)
    return styled(figure, height=max(260, 60 * layout.rows + 80))


def q_table_chart(layout: GridLayout, frame: PolicyFrame) -> go.Figure:
    """A tabela Q, uma grelha por ação, duas a duas: o que o agente espera de cada ação."""
    actions = len(layout.action_labels)
    rows = -(-actions // 2)
    figure = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=list(layout.action_labels),
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )
    for action in range(actions):
        row, column = divmod(action, 2)
        values = [
            None if chooses is None else q[action]
            for q, chooses in zip(frame.q_values, frame.actions, strict=True)
        ]
        figure.add_trace(
            go.Heatmap(
                z=_grid(layout, values),
                zmin=-1,
                zmax=1,
                colorscale=_VALUES,
                xgap=1,
                ygap=1,
                showscale=action == actions - 1,
                hovertemplate=(
                    f"{layout.action_labels[action]}: Q %{{z:.3f}} (linha %{{y}}, coluna %{{x}})"
                    "<extra></extra>"
                ),
            ),
            row=row + 1,
            col=column + 1,
        )
        # Células quadradas, como na grelha (cada grelha tem os seus eixos: x, x2, x3…).
        figure.update_yaxes(
            scaleanchor=f"x{action + 1 if action else ''}", row=row + 1, col=column + 1
        )
    figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, ticks="")
    figure.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, ticks="", autorange="reversed"
    )
    styled(figure, height=rows * min(260, 30 * layout.rows + 40) + 60)
    figure.update_layout(margin_t=40)  # o nome de cada ação fica por cima da sua grelha
    return figure


def _grid(layout: GridLayout, values: Sequence[float | None]) -> list[list[float | None]]:
    """Os valores por célula, arrumados em linhas (``None`` fica em branco)."""
    return [
        list(values[row * layout.columns : (row + 1) * layout.columns])
        for row in range(layout.rows)
    ]


def _grid_axes(figure: go.Figure, layout: GridLayout) -> None:
    """Eixos de um mapa: sem números, a linha 0 em cima e as células quadradas."""
    figure.update_xaxes(
        range=[-0.5, layout.columns - 0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        ticks="",
    )
    figure.update_yaxes(
        range=[layout.rows - 0.5, -0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        ticks="",
        scaleanchor="x",
    )
