"""Ecrã 10 · Laboratório: mexer num controlo e ver o efeito imediato num conceito."""

from typing import Literal

import streamlit as st

from charts.laboratory import (
    knn_accuracy_chart,
    log_options,
    loss_curves_chart,
    overfitting_errors_chart,
    overfitting_fit_chart,
    regularization_paths_chart,
    regularization_scores_chart,
    surface_chart,
)
from charts.training import animation_chart
from components.layout import render_header, render_why
from components.notices import render_notices
from components.state import get_lab

# As taxas de aprendizagem e os k que os seletores oferecem.
_RATES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
_KS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]


def main() -> None:
    """Desenha o ecrã."""
    render_header("laboratorio")
    overfitting, rate, knn, regularization = st.tabs(
        ["Overfitting", "Taxa de aprendizagem", "O k do KNN", "Regularização"]
    )
    with overfitting:
        _overfitting()
    with rate:
        _learning_rate()
    with knn:
        _knn()
    with regularization:
        _regularization()
    render_why("bias_variance", "learning_rate_demo", "knn_k", "regularization")


def _overfitting() -> None:
    st.markdown(
        "Uma regressão polinomial em 20 pontos de um seno com ruído. Sobe o grau e vê o erro de "
        "treino a descer sempre, enquanto o de teste desce e depois volta a subir."
    )
    degree = st.slider("Grau do polinómio", 1, 15, 3, key="laboratorio.degree")
    demo = get_lab().laboratory.overfitting(degree)
    left, right = st.columns(2)
    left.plotly_chart(overfitting_fit_chart(demo), width="stretch")
    right.plotly_chart(overfitting_errors_chart(demo), width="stretch")
    render_notices(demo.notices)


def _learning_rate() -> None:
    st.markdown(
        "A regressão linear com dois pesos, treinada por gradiente descendente durante 60 épocas. "
        "O erro é uma taça (as curvas de nível) e o caminho mostra os pesos época a época."
    )
    rate = st.select_slider(
        "Taxa de aprendizagem",
        options=_RATES,
        value=0.1,
        format_func=lambda value: f"{value:g}".replace(".", ","),
        key="laboratorio.rate",
    )
    demo = get_lab().laboratory.learning_rate(rate)
    left, right = st.columns(2)
    left.plotly_chart(surface_chart(demo), width="stretch")
    right.plotly_chart(loss_curves_chart(demo), width="stretch")
    render_notices(demo.notices)


def _knn() -> None:
    st.markdown(
        "O KNN (em NumPy) nas luas com ruído. Com poucos vizinhos a fronteira segue cada ponto; "
        "com muitos fica lisa demais. O melhor k está no meio."
    )
    k = st.select_slider("Número de vizinhos (k)", options=_KS, value=5, key="laboratorio.k")
    demo = get_lab().laboratory.knn(k)
    render_notices(demo.notices)
    # A fronteira precisa da largura toda (o plano é quadrado e tem a barra de cores).
    if demo.boundary is not None:
        st.plotly_chart(animation_chart(demo.boundary), width="stretch")
    st.plotly_chart(knn_accuracy_chart(demo), width="stretch")


def _regularization() -> None:
    st.markdown(
        "Uma regressão com 8 features em que só 3 contam (x1, x2 e x3) e só 40 linhas de treino. "
        "Vê os pesos a encolher à medida que o alpha cresce."
    )
    kind = st.segmented_control(
        "Modelo",
        ["ridge", "lasso"],
        default="lasso",
        required=True,
        format_func=lambda option: "Ridge (L2)" if option == "ridge" else "Lasso (L1)",
        key="laboratorio.kind",
    )
    model: Literal["ridge", "lasso"] = "ridge" if kind == "ridge" else "lasso"
    low, high = (0.01, 10_000.0) if model == "ridge" else (0.001, 100.0)
    options = log_options(low, high)
    alpha = st.select_slider(
        "Alpha (força da regularização)",
        options=options,
        value=options[len(options) // 2],
        format_func=lambda value: f"{value:g}".replace(".", ","),
        key=f"laboratorio.alpha.{model}",
    )
    demo = get_lab().laboratory.regularization(model, alpha)
    left, right = st.columns(2)
    left.plotly_chart(regularization_paths_chart(demo), width="stretch")
    right.plotly_chart(regularization_scores_chart(demo), width="stretch")
    render_notices(demo.notices)


main()
