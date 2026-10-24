"""Gráficos das demonstrações do ecrã Laboratório (os números vêm do ``LaboratoryService``)."""

import math

import plotly.graph_objects as go

from charts.distributions import CATEGORY_COLORS, styled
from ml_lab.core.services import KnnDemo, LearningRateDemo, OverfittingDemo, RegularizationDemo

_TRAIN, _TEST, _CHOSEN = CATEGORY_COLORS[0], CATEGORY_COLORS[1], CATEGORY_COLORS[3]


def overfitting_fit_chart(demo: OverfittingDemo) -> go.Figure:
    """Os pontos de treino e de teste, o seno verdadeiro e a curva do grau escolhido."""
    figure = go.Figure(
        [
            go.Scatter(
                x=list(demo.test.x),
                y=list(demo.test.y),
                mode="markers",
                name="Teste",
                marker={"size": 5, "color": _TEST, "opacity": 0.4},
            ),
            go.Scatter(
                x=list(demo.truth.x),
                y=list(demo.truth.y),
                mode="lines",
                name="Verdade (seno)",
                line={"color": "#888888", "dash": "dash"},
            ),
            go.Scatter(
                x=list(demo.curve.x),
                y=list(demo.curve.y),
                mode="lines",
                name=f"Grau {demo.degree}",
                line={"color": _CHOSEN, "width": 3},
            ),
            go.Scatter(
                x=list(demo.train.x),
                y=list(demo.train.y),
                mode="markers",
                name="Treino",
                marker={"size": 9, "color": _TRAIN, "line": {"color": "white", "width": 1}},
            ),
        ]
    )
    figure.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        yaxis_range=[-2.5, 2.5],
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=420)


def overfitting_errors_chart(demo: OverfittingDemo) -> go.Figure:
    """O erro no treino e no teste para cada grau, com o grau escolhido marcado."""
    figure = go.Figure(
        [
            go.Scatter(
                x=list(demo.degrees),
                y=list(demo.train_errors),
                mode="lines+markers",
                name="Treino",
                line={"color": _TRAIN},
            ),
            go.Scatter(
                x=list(demo.degrees),
                y=list(demo.test_errors),
                mode="lines+markers",
                name="Teste",
                line={"color": _TEST},
            ),
        ]
    )
    figure.add_vline(x=demo.degree, line={"color": _CHOSEN, "dash": "dash"})
    figure.update_layout(
        xaxis_title="Grau do polinómio",
        yaxis={"title": "Erro (RMSE)", "type": "log"},
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=420)


def surface_chart(demo: LearningRateDemo) -> go.Figure:
    """A taça do erro (curvas de nível) e o caminho dos pesos com a taxa escolhida."""
    figure = go.Figure(
        go.Contour(
            x=list(demo.surface_x),
            y=list(demo.surface_y),
            z=[list(row) for row in demo.surface],
            colorscale="Blues",
            reversescale=True,
            contours={"coloring": "heatmap", "showlines": True},
            ncontours=25,
            showscale=False,
            hovertemplate="w₁ %{x:.2f}, w₂ %{y:.2f}: erro %{z:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=list(demo.path.x),
            y=list(demo.path.y),
            mode="lines+markers",
            name=f"Caminho (taxa {demo.rate:g})".replace(".", ","),
            line={"color": _CHOSEN, "width": 2},
            marker={"size": 5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[demo.optimum[0]],
            y=[demo.optimum[1]],
            mode="markers",
            name="Fundo da taça",
            marker={"size": 14, "symbol": "x", "color": "#111111"},
        )
    )
    figure.update_layout(
        xaxis={"title": "Peso w₁", "range": [demo.surface_x[0], demo.surface_x[-1]]},
        yaxis={"title": "Peso w₂", "range": [demo.surface_y[0], demo.surface_y[-1]]},
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=460)


def loss_curves_chart(demo: LearningRateDemo) -> go.Figure:
    """A loss época a época com cada taxa (a escolhida em destaque)."""
    figure = go.Figure()
    for number, curve in enumerate(demo.curves):
        chosen = curve.rate == demo.rate
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(curve.losses) + 1)),
                y=list(curve.losses),
                mode="lines",
                name=f"Taxa {curve.rate:g}{' (explodiu)' if curve.diverged else ''}".replace(
                    ".", ","
                ),
                line={
                    "color": _CHOSEN if chosen else CATEGORY_COLORS[number % len(CATEGORY_COLORS)],
                    "width": 3 if chosen else 1.5,
                },
            )
        )
    figure.update_layout(
        xaxis_title="Época",
        yaxis={"title": "Loss (EQM)", "type": "log", "exponentformat": "power"},
        legend={"orientation": "h", "y": 1.15},
    )
    return styled(figure, height=460)


def knn_accuracy_chart(demo: KnnDemo) -> go.Figure:
    """A exatidão no treino e no teste para cada k (eixo logarítmico)."""
    figure = go.Figure(
        [
            go.Scatter(
                x=list(demo.ks),
                y=[value * 100 for value in demo.train_accuracy],
                mode="lines+markers",
                name="Treino",
                line={"color": _TRAIN},
            ),
            go.Scatter(
                x=list(demo.ks),
                y=[value * 100 for value in demo.test_accuracy],
                mode="lines+markers",
                name="Teste",
                line={"color": _TEST},
            ),
        ]
    )
    figure.add_vline(x=demo.k, line={"color": _CHOSEN, "dash": "dash"})
    figure.update_layout(
        xaxis={"title": "Número de vizinhos (k)", "type": "log"},
        yaxis_title="Exatidão (%)",
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=420)


def regularization_paths_chart(demo: RegularizationDemo) -> go.Figure:
    """O peso de cada feature à medida que o alpha cresce (as três verdadeiras a cheio)."""
    figure = go.Figure()
    for number, (name, path) in enumerate(zip(demo.feature_names, demo.paths, strict=True)):
        useful = demo.true_coefficients[number] != 0
        figure.add_trace(
            go.Scatter(
                x=list(demo.alphas),
                y=list(path),
                mode="lines",
                name=f"{name}{' (conta)' if useful else ''}",
                line={
                    "color": CATEGORY_COLORS[number % len(CATEGORY_COLORS)],
                    "width": 2.5 if useful else 1,
                    "dash": "solid" if useful else "dot",
                },
            )
        )
    figure.add_vline(x=demo.alpha, line={"color": "#888888", "dash": "dash"})
    figure.add_hline(y=0, line={"color": "#888888", "width": 1})
    figure.update_layout(
        xaxis={"title": "Alpha", "type": "log", "exponentformat": "power"},
        yaxis_title="Peso",
        legend={"orientation": "h", "y": -0.25},
    )
    return styled(figure, height=440)


def regularization_scores_chart(demo: RegularizationDemo) -> go.Figure:
    """O R² no treino e no teste para cada alpha."""
    figure = go.Figure(
        [
            go.Scatter(
                x=list(demo.alphas),
                y=list(demo.train_r2),
                mode="lines",
                name="Treino",
                line={"color": _TRAIN},
            ),
            go.Scatter(
                x=list(demo.alphas),
                y=list(demo.test_r2),
                mode="lines",
                name="Teste",
                line={"color": _TEST},
            ),
        ]
    )
    figure.add_vline(x=demo.alpha, line={"color": "#888888", "dash": "dash"})
    low = min([*demo.train_r2, *demo.test_r2, 0.0])
    figure.update_layout(
        xaxis={"title": "Alpha", "type": "log", "exponentformat": "power"},
        yaxis={"title": "R²", "range": [max(low, -1.0) - 0.05, 1.0]},
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=440)


def log_options(low: float, high: float, per_decade: int = 4) -> list[float]:
    """Valores "redondos" espaçados em escala logarítmica, para os seletores de alpha."""
    options = []
    for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        for step in (1.0, 2.5, 5.0, 7.5)[:per_decade]:
            value = float(f"{step * 10.0**exponent:.6g}")
            if low <= value <= high:
                options.append(value)
    return options
