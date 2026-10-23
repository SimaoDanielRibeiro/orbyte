"""Gráficos da avaliação: matriz de confusão e comparação entre treino e teste."""

import plotly.graph_objects as go

from charts.distributions import CATEGORY_COLORS, styled
from ml_lab.core.services import ConfusionMatrix, EvaluationReport


def confusion_matrix_chart(matrix: ConfusionMatrix) -> go.Figure:
    """Matriz de confusão: linhas são a classe real, colunas a prevista; a diagonal são acertos."""
    labels = list(matrix.labels)
    figure = go.Figure(
        go.Heatmap(
            z=[list(row) for row in matrix.counts],
            x=labels,
            y=labels,
            colorscale="Blues",
            texttemplate="%{z}",
            showscale=False,
            hovertemplate="real %{y}, previsto %{x}: %{z} exemplos<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Classe prevista",
        yaxis_title="Classe real",
        xaxis_type="category",
        yaxis_type="category",
        yaxis_autorange="reversed",
    )
    return styled(figure, height=max(320, 70 * len(labels) + 120))


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
