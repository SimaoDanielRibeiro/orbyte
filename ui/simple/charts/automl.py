"""Gráficos da procura dos agentes (ecrã Agentes): só desenham o que o serviço dá.

A cor diz só a ronda (triagem ou afinação): com sete famílias, sete cores num gráfico de
pontos confundem-se, sobretudo a quem vê mal as cores. A família de cada candidato está no
texto ao passar o rato e no gráfico "o melhor de cada agente", onde cada agente tem a sua
linha. As cores são as duas primeiras da paleta validada para daltonismo (a skill de
visualização de dados), que passam a verificação com todos os pares.

O objetivo em percentagem (a exatidão) desenha-se de 0 a 100; os outros (o R², a RMSE), tal
como vêm. Os textos ficam com a cor do tema (claro ou escuro), e as linhas de referência com
um cinzento médio, que se lê nos dois.
"""

from collections.abc import Sequence

import plotly.graph_objects as go

from charts.distributions import styled
from ml_lab.core.services import Candidate, ModelFamily, ObjectiveOption, SearchPhase
from ml_lab.formatting import format_value

_ROUNDS = {
    SearchPhase.SCREENING: ("Triagem", "#2a78d6", "circle"),
    SearchPhase.TUNING: ("Afinação", "#eb6834", "diamond"),
}
"""A legenda é curta (a explicação das rondas está por baixo do gráfico): assim, a barra de
ferramentas do Plotly, à direita, não a tapa."""

_GUIDE = "#8a8984"
"""As linhas de referência (o melhor até aí, a referência): não são uma série."""

_RING = {"width": 1.5, "color": "#ffffff"}
"""Um anel da cor do fundo, para os pontos sobrepostos não se fundirem."""


def search_chart(
    candidates: Sequence[Candidate], objective: ObjectiveOption, winner: int | None = None
) -> go.Figure:
    """Cada candidato pela ordem em que foi medido (com o desvio entre folds) e o melhor até aí."""
    measured = [c for c in candidates if c.mean is not None]
    figure = go.Figure()
    for phase, (name, color, symbol) in _ROUNDS.items():
        members = [c for c in measured if c.phase is phase]
        if not members:
            continue
        figure.add_trace(
            go.Scatter(
                x=[c.number for c in members],
                y=[_scaled(objective, c.mean) for c in members],
                error_y={
                    "type": "data",
                    "array": [_scaled(objective, c.std) or 0.0 for c in members],
                    "thickness": 1.5,
                    "width": 0,
                    "color": color,
                },
                mode="markers",
                name=name,
                marker={"size": 10, "color": color, "symbol": symbol, "line": _RING},
                customdata=[[c.model_display_name, c.family.label, _changes(c)] for c in members],
                hovertemplate=(
                    "candidato %{x} · %{customdata[0]}<br>%{customdata[1]}<br>"
                    f"%{{customdata[2]}}<br>{_value_format(objective)}<extra></extra>"
                ),
            )
        )
    steps, last = _running_best(candidates, objective)
    if steps:
        figure.add_trace(
            go.Scatter(
                x=[number for number, _ in steps],
                y=[value for _, value in steps],
                mode="lines",
                name="O melhor até aí",
                line={"shape": "hv", "color": _GUIDE, "width": 2, "dash": "dot"},
                hoverinfo="skip",
            )
        )
    tuning = [c.number for c in candidates if c.phase is SearchPhase.TUNING]
    if tuning:
        figure.add_vline(x=min(tuning) - 0.5, line={"color": _GUIDE, "width": 1, "dash": "dash"})
    chosen = next((c for c in measured if c.number == winner), None)
    if chosen is not None:
        figure.add_annotation(
            x=chosen.number,
            y=_scaled(objective, chosen.mean),
            text=f"vencedor: {chosen.model_display_name}",
            showarrow=True,
            arrowhead=0,
            arrowcolor=_GUIDE,
            ax=0,
            ay=-36,
        )
    figure.update_layout(
        xaxis={"title": "Candidato, pela ordem em que foi medido", "dtick": 1 if last <= 30 else 5},
        yaxis_title=f"{_axis_label(objective)} na validação cruzada",
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.14},
    )
    return styled(figure, height=380)


def agents_chart(
    candidates: Sequence[Candidate],
    objective: ObjectiveOption,
    baseline: Candidate | None = None,
) -> go.Figure:
    """O melhor candidato de cada agente (uma linha por família), com o desvio entre folds."""
    best: dict[ModelFamily, Candidate] = {}
    for candidate in candidates:
        if candidate.mean is None:
            continue
        current = best.get(candidate.family)
        if current is None or _better(objective, candidate.mean, current.mean):
            best[candidate.family] = candidate
    ordered = sorted(
        best.values(),
        key=lambda c: c.mean if c.mean is not None else 0.0,
        reverse=objective.higher_is_better,
    )
    figure = go.Figure(
        go.Scatter(
            x=[_scaled(objective, c.mean) for c in ordered],
            y=[c.family.label for c in ordered],
            error_x={
                "type": "data",
                "array": [_scaled(objective, c.std) or 0.0 for c in ordered],
                "thickness": 1.5,
                "width": 0,
                "color": _ROUNDS[SearchPhase.SCREENING][1],
            },
            mode="markers+text",
            marker={"size": 11, "color": _ROUNDS[SearchPhase.SCREENING][1], "line": _RING},
            text=[c.model_display_name for c in ordered],
            textposition="top center",
            textfont={"size": 11},
            customdata=[[c.model_display_name, _changes(c)] for c in ordered],
            hovertemplate=(
                f"%{{y}}<br>%{{customdata[0]}} · %{{customdata[1]}}<br>"
                f"{_value_format(objective, axis='x')}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    if baseline is not None and baseline.mean is not None:
        figure.add_vline(
            x=_scaled(objective, baseline.mean),
            line={"color": _GUIDE, "width": 1, "dash": "dash"},
            annotation_text=f"referência: {baseline.model_display_name}",
            annotation_position="bottom right",
            annotation_font={"size": 11},
        )
    figure.update_layout(
        xaxis_title=f"{_axis_label(objective)} na validação cruzada (o melhor de cada agente)",
        yaxis={"autorange": "reversed", "title": None},
    )
    return styled(figure, height=max(260, 56 * len(ordered) + 90))


def _running_best(
    candidates: Sequence[Candidate], objective: ObjectiveOption
) -> tuple[list[tuple[int, float]], int]:
    """Os degraus do melhor até aí (candidato, valor) e o número do último candidato."""
    steps: list[tuple[int, float]] = []
    best: float | None = None
    for candidate in candidates:
        if candidate.mean is not None and (
            best is None or _better(objective, candidate.mean, best)
        ):
            best = candidate.mean
            steps.append((candidate.number, _scaled(objective, best) or 0.0))
    last = candidates[-1].number if candidates else 0
    if steps and steps[-1][0] != last:
        steps.append((last, steps[-1][1]))  # o degrau estende-se até ao último candidato
    return steps, last


def _better(objective: ObjectiveOption, value: float, than: float | None) -> bool:
    if than is None:
        return True
    return value > than if objective.higher_is_better else value < than


def _scaled(objective: ObjectiveOption, value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100 if objective.as_percent else value


def _axis_label(objective: ObjectiveOption) -> str:
    return f"{objective.label} (%)" if objective.as_percent else objective.label


def _value_format(objective: ObjectiveOption, axis: str = "y") -> str:
    if objective.as_percent:
        return f"%{{{axis}:.1f}}%"
    return f"%{{{axis}:.3f}}"


def _changes(candidate: Candidate) -> str:
    """Os hiperparâmetros que o agente mudou, com os números à portuguesa."""
    if not candidate.changes:
        return "valores por omissão"
    return ", ".join(f"{name} = {format_value(value)}" for name, value in candidate.changes.items())
