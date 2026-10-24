"""Gráficos da avaliação e da explicabilidade (ecrãs Avaliar e Explicar).

Só desenham: os números chegam prontos dos serviços (``EvaluationCharts``, validação
cruzada, importância, dependência parcial), para a interface aprimorada desenhar os
mesmos.
"""

from collections.abc import Sequence
from itertools import pairwise

import plotly.graph_objects as go

from charts.distributions import CATEGORY_COLORS, PRIMARY, styled
from ml_lab.core.services import (
    ConfusionMatrix,
    CrossValidation,
    Curve,
    Dendrogram,
    Distribution,
    EvaluationReport,
    LearningCurve,
    PartialDependence,
    PermutationImportance,
    PointCloud,
    Residuals,
    RowExplanation,
    Silhouette,
)

# As cores de "certo" e "errado", e as das partes dos dados (treino e teste).
_GOOD, _BAD = CATEGORY_COLORS[2], CATEGORY_COLORS[3]
_TRAIN, _TEST = CATEGORY_COLORS[0], CATEGORY_COLORS[1]
_NOISE = "#9E9E9E"
_MARKS = {"Certo": _GOOD, "Errado": _BAD, "Normal": PRIMARY, "Anomalia": _BAD, "Ruído": _NOISE}


def confusion_matrix_chart(matrix: ConfusionMatrix) -> go.Figure:
    """Matriz de confusão (ou tabela cruzada): linhas reais, colunas previstas (ou grupos)."""
    rows, columns = list(matrix.labels), list(matrix.columns)
    same = rows == columns
    figure = go.Figure(
        go.Heatmap(
            z=[list(row) for row in matrix.counts],
            x=columns,
            y=rows,
            colorscale="Blues",
            texttemplate="%{z}",
            showscale=False,
            hovertemplate="real %{y}, previsto %{x}: %{z} exemplos<extra></extra>"
            if same
            else "%{y} no %{x}: %{z} pontos<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Classe prevista" if same else "Grupo encontrado",
        yaxis_title="Classe real" if same else "Referência",
        xaxis_type="category",
        yaxis_type="category",
        yaxis_autorange="reversed",
    )
    return styled(figure, height=max(320, 70 * len(rows) + 120))


def train_test_chart(train: EvaluationReport, test: EvaluationReport) -> go.Figure:
    """Barras lado a lado com as métricas em percentagem no treino e no teste."""
    names = [m.name for m in train.metrics if m.as_percent and m.value is not None]
    figure = go.Figure()
    for color, report in zip(CATEGORY_COLORS, (train, test), strict=False):
        results = [report.metric(name) for name in names]
        figure.add_trace(
            go.Bar(
                name=report.partition.label,
                x=[result.display_name for result in results],
                y=[(result.value or 0.0) * 100 for result in results],
                marker_color=color,
                hovertemplate="%{x}: %{y:.1f}%<extra>" + report.partition.label + "</extra>",
            )
        )
    figure.update_layout(barmode="group", yaxis_title="%", yaxis_range=[0, 100])
    return styled(figure)


def curves_chart(curves: Sequence[Curve], *, kind: str) -> go.Figure:
    """Curvas ROC (``kind="roc"``) ou de precisão-recall (``kind="pr"``), uma por classe."""
    figure = go.Figure()
    for number, curve in enumerate(curves):
        area = "AUC" if kind == "roc" else "AP"
        figure.add_trace(
            go.Scatter(
                x=list(curve.x),
                y=list(curve.y),
                mode="lines",
                name=f"{curve.label} ({area} {curve.area:.3f})".replace(".", ","),
                line={"color": CATEGORY_COLORS[number % len(CATEGORY_COLORS)], "width": 2},
            )
        )
    if kind == "roc":
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ao acaso",
                line={"color": "#888888", "dash": "dash", "width": 1},
            )
        )
        figure.update_layout(
            xaxis_title="Falsos positivos (1 − especificidade)", yaxis_title="Recall"
        )
    else:
        figure.update_layout(xaxis_title="Recall", yaxis_title="Precisão")
    figure.update_layout(
        xaxis_range=[0, 1], yaxis_range=[0, 1.02], legend={"orientation": "h", "y": -0.25}
    )
    return styled(figure, height=420)


def residuals_chart(residuals: Residuals) -> go.Figure:
    """O erro (real − previsto) de cada linha contra o valor previsto: deve parecer ruído."""
    figure = go.Figure(
        go.Scattergl(
            x=list(residuals.predicted),
            y=list(residuals.errors),
            mode="markers",
            marker={"size": 6, "color": PRIMARY, "opacity": 0.6},
            hovertemplate="previsto %{x:.3~f}, erro %{y:.3~f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line={"color": "#888888", "dash": "dash"})
    figure.update_layout(xaxis_title="Valor previsto", yaxis_title="Erro (real − previsto)")
    return styled(figure, height=380)


def actual_vs_predicted_chart(residuals: Residuals) -> go.Figure:
    """O valor real contra o previsto: um modelo perfeito põe tudo na diagonal."""
    low = min([*residuals.actual, *residuals.predicted], default=0.0)
    high = max([*residuals.actual, *residuals.predicted], default=1.0)
    figure = go.Figure(
        go.Scattergl(
            x=list(residuals.predicted),
            y=list(residuals.actual),
            mode="markers",
            marker={"size": 6, "color": PRIMARY, "opacity": 0.6},
            hovertemplate="previsto %{x:.3~f}, real %{y:.3~f}<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": "#888888", "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.update_layout(xaxis_title="Valor previsto", yaxis_title="Valor real")
    return styled(figure, height=380)


def distribution_chart(
    distribution: Distribution, *, x_title: str, threshold: float | None = None
) -> go.Figure:
    """Histograma (empilhado, se houver várias séries) com um limiar opcional."""
    edges = distribution.edges
    centers = [(low + high) / 2 for low, high in pairwise(edges)]
    widths = [high - low for low, high in pairwise(edges)]
    figure = go.Figure()
    for number, (name, counts) in enumerate(
        zip(distribution.series, distribution.counts, strict=True)
    ):
        figure.add_trace(
            go.Bar(
                x=centers,
                y=list(counts),
                width=widths,
                name=name,
                marker_color=_MARKS.get(name, CATEGORY_COLORS[number % len(CATEGORY_COLORS)]),
            )
        )
    if threshold is not None:
        figure.add_vline(
            x=threshold,
            line={"color": "#888888", "dash": "dash"},
            annotation_text="limiar",
            annotation_position="top",
        )
    figure.update_layout(
        barmode="stack",
        xaxis_title=x_title,
        yaxis_title="Pontos",
        showlegend=len(distribution.series) > 1,
    )
    return styled(figure, height=340)


def plane_chart(
    points: PointCloud, axis_labels: tuple[str, str], *, value_title: str = ""
) -> go.Figure:
    """Os pontos no plano 2D, pintados pela etiqueta (certo/errado, grupo…) ou por um valor."""
    figure = go.Figure()
    if points.labels:
        names = list(dict.fromkeys(points.labels))
        groups = sorted(names, key=lambda name: (name == "Ruído", name))
        for number, label in enumerate(groups):
            chosen = [i for i, value in enumerate(points.labels) if value == label]
            figure.add_trace(
                go.Scattergl(
                    x=[points.x[i] for i in chosen],
                    y=[points.y[i] for i in chosen],
                    mode="markers",
                    name=label,
                    marker={
                        "size": 7,
                        "color": _MARKS.get(label, CATEGORY_COLORS[number % len(CATEGORY_COLORS)]),
                    },
                )
            )
    else:
        figure.add_trace(
            go.Scattergl(
                x=list(points.x),
                y=list(points.y),
                mode="markers",
                marker={
                    "size": 7,
                    "color": list(points.values),
                    "colorscale": "OrRd",
                    "showscale": True,
                    "colorbar": {"title": {"text": value_title}},
                },
                hovertemplate=f"{value_title.lower()} %{{marker.color:.3~f}}<extra></extra>",
                showlegend=False,
            )
        )
    x_title, y_title = axis_labels
    figure.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return styled(figure, height=460)


def silhouette_chart(silhouette: Silhouette) -> go.Figure:
    """O perfil de silhueta: as barras de cada grupo, ordenadas, e a média geral."""
    figure = go.Figure()
    offset = 0
    for number, group in enumerate(silhouette.groups):
        positions = list(range(offset, offset + len(group.values)))
        figure.add_trace(
            go.Bar(
                x=list(group.values),
                y=positions,
                orientation="h",
                name=group.label,
                marker_color=CATEGORY_COLORS[number % len(CATEGORY_COLORS)],
                marker_line_width=0,
                hoverinfo="skip",
            )
        )
        offset += len(group.values) + 5  # um espaço entre grupos
    figure.add_vline(
        x=silhouette.average,
        line={"color": "#888888", "dash": "dash"},
        annotation_text=f"média {silhouette.average:.2f}".replace(".", ","),
    )
    figure.update_layout(
        bargap=0,
        xaxis_title="Silhueta de cada ponto",
        yaxis={"showticklabels": False, "title": "Pontos, por grupo"},
        xaxis_range=[min(-0.1, min((min(g.values) for g in silhouette.groups), default=0)), 1],
    )
    return styled(figure, height=460)


def dendrogram_chart(dendrogram: Dendrogram) -> go.Figure:
    """O dendrograma: cada junção é um "U" invertido à altura da distância a que se deu."""
    figure = go.Figure()
    for segment in dendrogram.segments:
        figure.add_trace(
            go.Scatter(
                x=[x for x, _ in segment],
                y=[y for _, y in segment],
                mode="lines",
                line={"color": PRIMARY, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if dendrogram.cut_height is not None:
        figure.add_hline(
            y=dendrogram.cut_height,
            line={"color": _BAD, "dash": "dash"},
            annotation_text="corte",
        )
    figure.update_layout(
        xaxis={
            "tickvals": list(dendrogram.leaf_positions),
            "ticktext": list(dendrogram.leaf_labels),
            "title": "Folhas (entre parênteses, quantos pontos juntam)",
        },
        yaxis_title="Distância da junção",
    )
    return styled(figure, height=420)


def bars_chart(
    pairs: Sequence[tuple[str, float]], *, x_title: str, percent: bool = False
) -> go.Figure:
    """Barras simples, de cima para baixo pela ordem recebida."""
    labels = [label for label, _ in pairs][::-1]
    values = [value * 100 if percent else value for _, value in pairs][::-1]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=PRIMARY,
            hovertemplate="%{y}: %{x:.3~f}" + ("%" if percent else "") + "<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title=x_title, yaxis={"automargin": True})
    return styled(figure, height=max(260, 28 * len(pairs) + 80))


def cross_validation_chart(validation: CrossValidation) -> go.Figure:
    """A métrica principal em cada fold, no treino desse fold e no fold deixado de fora."""
    scale = 100 if validation.as_percent else 1
    folds = [f"Fold {fold.fold}" for fold in validation.folds]
    figure = go.Figure()
    for name, color, values in (
        ("Treino do fold", _TRAIN, [fold.train for fold in validation.folds]),
        ("Fold de fora", _TEST, [fold.test for fold in validation.folds]),
    ):
        figure.add_trace(
            go.Bar(
                x=folds,
                y=[None if value is None else value * scale for value in values],
                name=name,
                marker_color=color,
            )
        )
    if validation.mean is not None:
        figure.add_hline(
            y=validation.mean * scale,
            line={"color": "#888888", "dash": "dash"},
            annotation_text="média",
        )
    figure.update_layout(
        barmode="group",
        yaxis_title=f"{validation.metric_label}{' (%)' if validation.as_percent else ''}",
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=360)


def learning_curve_chart(curve: LearningCurve) -> go.Figure:
    """A métrica no treino e na avaliação para cada tamanho do treino, com o desvio padrão."""
    scale = 100 if curve.as_percent else 1
    sizes = [point.size for point in curve.points]
    figure = go.Figure()
    for name, color, means, stds in (
        (
            "Treino",
            _TRAIN,
            [p.train_mean for p in curve.points],
            [p.train_std for p in curve.points],
        ),
        (
            "Avaliação",
            _TEST,
            [p.test_mean for p in curve.points],
            [p.test_std for p in curve.points],
        ),
    ):
        upper = [(m + s) * scale for m, s in zip(means, stds, strict=True)]
        lower = [(m - s) * scale for m, s in zip(means, stds, strict=True)]
        figure.add_trace(
            go.Scatter(
                x=[*sizes, *sizes[::-1]],
                y=[*upper, *lower[::-1]],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=sizes,
                y=[m * scale for m in means],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
            )
        )
    figure.update_layout(
        xaxis_title="Linhas de treino",
        yaxis_title=f"{curve.metric_label}{' (%)' if curve.as_percent else ''}",
        legend={"orientation": "h", "y": 1.12},
    )
    return styled(figure, height=360)


def importance_chart(importance: PermutationImportance) -> go.Figure:
    """Quanto a métrica principal piora quando se baralha cada feature (com o desvio padrão)."""
    scale = 100 if importance.as_percent else 1
    items = list(importance.importances)[::-1]
    figure = go.Figure(
        go.Bar(
            x=[item.mean * scale for item in items],
            y=[item.feature for item in items],
            orientation="h",
            error_x={"type": "data", "array": [item.std * scale for item in items]},
            marker_color=[PRIMARY if item.mean > 0 else _NOISE for item in items],
            hovertemplate="%{y}: %{x:.3~f}<extra></extra>",
        )
    )
    unit = " (pontos percentuais)" if importance.as_percent else ""
    figure.update_layout(
        xaxis_title=f"Quanto piora: {importance.metric_label}{unit}",
        yaxis={"automargin": True},
    )
    return styled(figure, height=max(300, 26 * len(items) + 100))


def dependence_chart(dependence: PartialDependence) -> go.Figure:
    """A previsão média ao longo dos valores da feature, com as curvas individuais (ICE)."""
    figure = go.Figure()
    grid = list(dependence.grid)
    for number, curve in enumerate(dependence.individual):
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=list(curve),
                mode="lines",
                line={"color": "#999999", "width": 1},
                opacity=0.35,
                name="Linhas individuais (ICE)",
                showlegend=number == 0,
                hoverinfo="skip",
            )
        )
    for number, (label, average) in enumerate(
        zip(dependence.output_labels, dependence.average, strict=True)
    ):
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=list(average),
                mode="lines",
                name=f"{label} (média)",
                line={"color": CATEGORY_COLORS[number % len(CATEGORY_COLORS)], "width": 3},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=list(dependence.values),
            y=[0] * len(dependence.values),
            mode="markers",
            marker={"symbol": "line-ns-open", "size": 10, "color": "#666666"},
            yaxis="y2",
            name="Onde há dados",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        xaxis_title=f"{dependence.feature} (depois da preparação)",
        yaxis_title="Previsão",
        yaxis2={"overlaying": "y", "range": [0, 20], "visible": False},
        legend={"orientation": "h", "y": -0.25},
    )
    return styled(figure, height=440)


def contributions_chart(explanation: RowExplanation, *, limit: int = 15) -> go.Figure:
    """O contributo de cada feature para uma previsão (as mais fortes, em valor absoluto)."""
    items = list(explanation.contributions[:limit])[::-1]
    figure = go.Figure(
        go.Bar(
            x=[item.effect for item in items],
            y=[f"{item.feature} = {item.value:.2f}".replace(".", ",") for item in items],
            orientation="h",
            marker_color=[_GOOD if item.effect > 0 else _BAD for item in items],
            hovertemplate="%{y}: %{x:.3~f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line={"color": "#888888"})
    figure.update_layout(xaxis_title=explanation.effect_label, yaxis={"automargin": True})
    return styled(figure, height=max(300, 26 * len(items) + 100))
