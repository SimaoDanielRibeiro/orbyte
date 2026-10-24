"""Ecrã 6 · Avaliar: métricas, gráficos de cada tarefa, diagnósticos e o que o modelo aprendeu."""

from typing import Any

import streamlit as st

from charts.evaluation import (
    actual_vs_predicted_chart,
    bars_chart,
    confusion_matrix_chart,
    cross_validation_chart,
    curves_chart,
    dendrogram_chart,
    distribution_chart,
    learning_curve_chart,
    plane_chart,
    residuals_chart,
    silhouette_chart,
    train_test_chart,
)
from charts.training import embedding_chart
from components.layout import render_header, render_why
from components.notices import render_error, render_notices
from components.state import get_lab, last_result
from ml_lab.core.services import (
    EvaluationCharts,
    EvaluationReport,
    ExperimentResult,
    MLLabError,
    Partition,
    TaskType,
)
from ml_lab.formatting import format_number, format_percent

# Quantas métricas aparecem em destaque, no topo. Num ecrã, um texto solto (mesmo uma
# docstring de constante) é desenhado pela "magia" do Streamlit: daí o comentário.
_CARDS = 4

# O nome a mostrar das propriedades simples do estado dos modelos.
_STATE_LABELS = {
    "iterations": "Iterações (ou épocas) do treino",
    "converged": "Convergiu",
    "inertia": "Inércia (soma das distâncias ao quadrado)",
    "n_support": "Vetores de suporte",
    "n_clusters": "Grupos",
    "n_noise": "Pontos de ruído",
    "n_core": "Pontos núcleo",
    "depth": "Profundidade da árvore",
    "n_leaves": "Folhas",
    "n_trees": "Árvores",
    "n_stages": "Etapas",
    "oob_score": "Pontuação fora do saco (OOB)",
    "threshold": "Limiar de anomalia",
    "kl_divergence": "Divergência KL final",
    "log_likelihood": "Log-verosimilhança média",
    "n_terms": "Termos do polinómio",
    "degree": "Grau",
    "n_train": "Exemplos guardados",
}


def main() -> None:
    """Desenha o ecrã."""
    render_header("avaliar")
    result = last_result()
    if result is None:
        st.info("Ainda não treinaste nenhum modelo.", icon=":material/model_training:")
        st.page_link(
            "pages/modelo.py", label="Ir para o ecrã Modelo", icon=":material/arrow_forward:"
        )
        return
    st.markdown(
        f"**{result.model_display_name}** em **{result.dataset_name}** · "
        f"{result.task_type.label.lower()} · seed {result.seed} · "
        f"{format_number(result.duration_seconds, 2)} s · run `{result.run_id}`"
    )
    report = _main_report(result)
    if report.partition is Partition.TEST:
        st.caption("No teste, com linhas que o modelo nunca viu:")
    else:
        st.caption("No treino (este modelo só coloca no mapa os pontos com que treinou):")
    columns = st.columns(_CARDS)
    for column, metric in zip(columns, report.metrics[:_CARDS], strict=False):
        column.metric(metric.display_name, metric.formatted, help=metric.what, border=True)
    render_notices([*result.notices, *report.notices], notes_title="Leitura dos resultados")

    charts = _charts(result)
    tabs = _tabs(result, charts)
    for name, tab in zip(tabs, st.tabs(list(tabs)), strict=True):
        with tab:
            tabs[name]()
    with st.expander("Configuração do run (para o repetir)", icon=":material/settings:"):
        st.json(result.config.model_dump(mode="json"), expanded=False)
    render_why(*_topics(result.task_type))


def _main_report(result: ExperimentResult) -> EvaluationReport:
    """O relatório do teste ou, se ele não tiver métricas (t-SNE), o do treino."""
    test = result.reports.get(Partition.TEST)
    if test is not None and test.metrics:
        return test
    return result.reports[Partition.TRAIN]


def _charts(result: ExperimentResult) -> EvaluationCharts | None:
    try:
        return get_lab().analysis.charts(result.run_id)
    except MLLabError:  # o run saiu da memória (só ficam os últimos)
        return None


def _tabs(result: ExperimentResult, charts: EvaluationCharts | None) -> dict[str, Any]:
    """Os separadores do ecrã, conforme a tarefa (cada um é uma função que o desenha)."""
    task = result.task_type
    tabs: dict[str, Any] = {"Métricas": lambda: _metrics(result)}
    if charts is None:
        tabs["Gráficos"] = _gone
    elif task.is_classification:
        tabs["Matriz de confusão"] = lambda: _confusion(result)
        tabs["Curvas ROC e PR"] = lambda: _curves(charts)
        tabs["Onde erra"] = lambda: _plane(charts, "Certo ou errado no teste")
    elif task is TaskType.REGRESSION:
        tabs["Resíduos"] = lambda: _residuals(charts)
        tabs["Onde erra"] = lambda: _plane(charts, "Erro absoluto")
    elif task is TaskType.CLUSTERING:
        tabs["Grupos"] = lambda: _groups(charts)
        tabs["Silhueta"] = lambda: _silhouette(charts)
        if charts.dendrogram is not None:
            tabs["Dendrograma"] = lambda: _dendrogram(charts)
        if _has_matrix(result):
            tabs["Referência"] = lambda: _confusion(result)
    elif task is TaskType.DIMENSIONALITY_REDUCTION:
        tabs["Mapa"] = lambda: _map(result)
        if charts.explained_variance:
            tabs["Variância explicada"] = lambda: _variance(charts)
    else:
        tabs["Pontuações"] = lambda: _scores(charts)
        tabs["Anomalias no plano"] = lambda: _plane(charts, "Anomalias no teste")
        if _has_matrix(result):
            tabs["Referência"] = lambda: _confusion(result)
    if task is not TaskType.DIMENSIONALITY_REDUCTION:
        tabs["Treino vs. teste"] = lambda: _compare(result)
    if task.requires_target:
        tabs["Validação cruzada"] = lambda: _validation(result)
    tabs["O que o modelo aprendeu"] = lambda: _learned(result, charts)
    return tabs


def _topics(task: TaskType) -> tuple[str, ...]:
    if task.is_classification:
        return ("roc_curve", "cross_validation", "learning_curve")
    if task is TaskType.REGRESSION:
        return ("residuals", "cross_validation", "learning_curve")
    if task is TaskType.CLUSTERING:
        return ("silhouette", "reference")
    return ("reference",)


def _gone() -> None:
    st.info(
        "Os gráficos precisam do modelo, e este run já não está em memória (só ficam os "
        "últimos). Treina-o outra vez para os ver.",
        icon=":material/history:",
    )


def _metrics(result: ExperimentResult) -> None:
    report = _main_report(result)
    train = result.reports[Partition.TRAIN]
    for metric in report.metrics:
        with st.expander(f"{metric.display_name}: {metric.formatted}"):
            st.markdown(f"**O que mede.** {metric.what}")
            st.markdown(f"**Como se lê.** {metric.how_to_read}")
            if report is not train:
                found = train.find(metric.name)
                st.caption(f"No treino: {found.formatted if found else '—'}.")


def _has_matrix(result: ExperimentResult) -> bool:
    return any(report.confusion_matrix is not None for report in result.reports.values())


def _confusion(result: ExperimentResult) -> None:
    parts = [part for part, report in result.reports.items() if report.confusion_matrix]
    part = st.segmented_control(
        "Parte dos dados",
        parts,
        default=Partition.TEST if Partition.TEST in parts else parts[0],
        required=True,
        format_func=lambda option: option.label,
        key="avaliar.partition",
    )
    matrix = result.reports[part or parts[0]].confusion_matrix
    assert matrix is not None  # só se mostram as partes que a têm
    st.plotly_chart(confusion_matrix_chart(matrix), width="stretch")
    if result.task_type is TaskType.CLUSTERING:
        st.caption(
            "Cada linha é uma classe da coluna de referência e cada coluna um grupo encontrado: "
            "grupos que correspondem a classes têm quase tudo numa só célula da coluna."
        )
    elif result.task_type is TaskType.ANOMALY_DETECTION:
        st.caption(
            "Cada linha é o que o ponto era (segundo a referência) e cada coluna o que o detetor "
            "disse: a diagonal são os acertos."
        )
    else:
        st.caption(
            "Cada linha é a classe real e cada coluna a prevista: a diagonal são os acertos, o "
            "resto são as confusões do modelo."
        )


def _curves(charts: EvaluationCharts) -> None:
    if not charts.roc:
        st.info(
            "Este modelo não dá probabilidades (ou falta uma classe no teste): sem elas, não há "
            "curvas.",
            icon=":material/info:",
        )
        return
    roc, pr = st.columns(2)
    roc.plotly_chart(curves_chart(charts.roc, kind="roc"), width="stretch")
    pr.plotly_chart(curves_chart(charts.pr, kind="pr"), width="stretch")
    st.caption(
        "Cada ponto de uma curva é um limiar de decisão diferente. A ROC compara os acertos com "
        "os falsos alarmes (a diagonal é o acaso); a de precisão-recall mostra o preço de "
        "encontrar mais casos. Em multiclasse, há uma curva por classe contra as outras."
    )


def _plane(charts: EvaluationCharts, title: str) -> None:
    if charts.test_points is None:
        st.info("Os dados não têm uma projeção 2D para desenhar.", icon=":material/info:")
        return
    st.plotly_chart(
        plane_chart(charts.test_points, charts.axis_labels, value_title=title), width="stretch"
    )
    st.caption(
        f"{title}, no plano da projeção 2D do ecrã Preparar. Zonas com muitos erros mostram "
        "onde o modelo tem mais dificuldade (muitas vezes, onde as classes se misturam)."
    )


def _residuals(charts: EvaluationCharts) -> None:
    residuals = charts.residuals
    if residuals is None:
        return
    left, right = st.columns(2)
    left.plotly_chart(residuals_chart(residuals), width="stretch")
    right.plotly_chart(actual_vs_predicted_chart(residuals), width="stretch")
    st.plotly_chart(
        distribution_chart(residuals.histogram, x_title="Erro (real − previsto)"), width="stretch"
    )
    st.caption(
        "Num bom modelo, os erros parecem ruído à volta de zero, sem padrão. Uma curva nos "
        "resíduos indica uma relação que o modelo não apanhou; um leque a abrir, um erro que "
        "cresce com o valor previsto."
    )


def _groups(charts: EvaluationCharts) -> None:
    left, right = st.columns([2, 3])
    if charts.cluster_sizes:
        left.plotly_chart(
            bars_chart(charts.cluster_sizes, x_title="Pontos de teste"), width="stretch"
        )
    if charts.test_points is not None:
        right.plotly_chart(plane_chart(charts.test_points, charts.axis_labels), width="stretch")
    st.caption(
        "O tamanho de cada grupo e onde ficam os pontos de teste no plano 2D. Grupos muito "
        "desequilibrados (um enorme e vários minúsculos) costumam querer dizer um k errado."
    )


def _silhouette(charts: EvaluationCharts) -> None:
    if charts.silhouette is None:
        st.info("A silhueta precisa de pelo menos dois grupos com pontos.", icon=":material/info:")
        return
    st.plotly_chart(silhouette_chart(charts.silhouette), width="stretch")
    st.caption(
        "Cada barra é um ponto: perto de 1, está bem no seu grupo; perto de 0, está na "
        "fronteira; negativa, estaria melhor noutro. A linha tracejada é a média."
    )


def _dendrogram(charts: EvaluationCharts) -> None:
    assert charts.dendrogram is not None  # o separador só existe quando há dendrograma
    st.plotly_chart(dendrogram_chart(charts.dendrogram), width="stretch")
    st.caption(
        "Cada «U» é uma junção de dois grupos, à altura da distância a que se deu (só as "
        "últimas 30). A linha vermelha é o corte que dá os grupos escolhidos: uma junção muito "
        "mais alta do que as de baixo é um bom sítio para cortar."
    )


def _map(result: ExperimentResult) -> None:
    if result.embedding is None:
        return
    st.plotly_chart(embedding_chart(result.embedding), width="stretch")
    st.caption(
        "Os pontos de treino nas novas coordenadas, pintados com a coluna de referência (se a "
        "escolheste). A confiança dos vizinhos diz se o que está junto no mapa estava junto nos "
        "dados."
    )


def _variance(charts: EvaluationCharts) -> None:
    ratios = charts.explained_variance
    total = 0.0
    pairs = []
    for number, ratio in enumerate(ratios, start=1):
        total += ratio
        pairs.append((f"Componente {number} (acumulado {format_percent(total)})", ratio))
    st.plotly_chart(
        bars_chart(pairs, x_title="Variância explicada (%)", percent=True), width="stretch"
    )
    st.caption(
        f"As componentes guardadas explicam {format_percent(total)} da variância dos dados. O "
        "que falta perdeu-se na redução."
    )


def _scores(charts: EvaluationCharts) -> None:
    if charts.scores is not None:
        st.plotly_chart(
            distribution_chart(
                charts.scores, x_title="Pontuação de anomalia", threshold=charts.threshold
            ),
            width="stretch",
        )
        st.caption(
            "A pontuação de cada ponto de teste, pintada com a coluna de referência (as anomalias "
            "verdadeiras). Acima do limiar, o detetor marca o ponto como anomalia: as anomalias à "
            "esquerda da linha escaparam-lhe e os pontos normais à direita são falsos alarmes."
            if charts.scores_reference
            else "A pontuação de cada ponto de teste, pintada com o que o detetor decidiu. Acima "
            "do limiar, o ponto é marcado como anomalia."
        )
    if charts.roc:
        st.plotly_chart(curves_chart(charts.roc, kind="roc"), width="stretch")
        st.caption(
            "A curva ROC da pontuação contra as anomalias verdadeiras (da coluna de referência): "
            "não depende do limiar."
        )


def _compare(result: ExperimentResult) -> None:
    test = result.reports.get(Partition.TEST)
    train = result.reports[Partition.TRAIN]
    if test is None or not test.metrics:
        st.info("Este modelo só tem relatório do treino.", icon=":material/info:")
        return
    if any(metric.as_percent and metric.value is not None for metric in test.metrics):
        st.plotly_chart(train_test_chart(train, test), width="stretch")
    table: dict[str, list[str]] = {"Métrica": [m.display_name for m in test.metrics]}
    for part, report in result.reports.items():
        table[part.label] = [
            (found.formatted if (found := report.find(m.name)) else "—") for m in test.metrics
        ]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "O modelo aprendeu com o treino; o teste mostra como se sai com dados que nunca viu. "
        "Uma diferença grande a favor do treino é o sinal de overfitting."
    )


def _validation(result: ExperimentResult) -> None:
    key = f"avaliar.validation.{result.run_id}"
    st.markdown(
        "A validação cruzada volta a treinar o modelo 5 vezes, uma por fold do treino (o teste "
        "fica de fora), e a curva de aprendizagem treina-o com cada vez mais linhas. Pode "
        "demorar."
    )
    if st.button("Calcular", icon=":material/play_arrow:", key=f"{key}.run"):
        st.session_state[key] = True
    if not st.session_state.get(key):
        return
    analysis = get_lab().analysis
    try:
        with st.spinner("A treinar nos folds…"):
            validation = analysis.cross_validation(result.run_id)
            curve = analysis.learning_curve(result.run_id)
    except MLLabError as error:
        render_error(error)
        return
    left, right = st.columns(2)
    left.plotly_chart(cross_validation_chart(validation), width="stretch")
    right.plotly_chart(learning_curve_chart(curve), width="stretch")
    render_notices([*validation.notices, *curve.notices], notes_title="Leitura")


def _learned(result: ExperimentResult, charts: EvaluationCharts | None) -> None:
    state = result.model_state
    shown = _coefficients(result)
    importances = state.get("feature_importances")
    if importances and len(importances) == len(result.feature_names):
        pairs = sorted(
            zip(result.feature_names, importances, strict=True), key=lambda pair: -pair[1]
        )[:20]
        st.markdown("**Importância de cada feature (pelas impurezas das árvores)**")
        st.plotly_chart(bars_chart(pairs, x_title="Importância"), width="stretch")
        st.caption(
            "Quanto cada feature reduziu a impureza nas divisões das árvores. Favorece features "
            "com muitos valores: a importância por permutação (ecrã Explicar) é mais honesta."
        )
        shown = True
    if charts is not None and charts.tree_rules:
        st.markdown("**As regras da árvore** (limiares na escala depois da preparação)")
        st.code("\n".join(charts.tree_rules), language=None)
        shown = True
    shown = _rows_table(result, "centers", "Centros (médias) de cada grupo ou classe") or shown
    shown = _rows_table(result, "components", "Peso de cada feature em cada componente") or shown
    scalars = {
        _STATE_LABELS.get(name, name): _written(value)
        for name, value in state.items()
        if isinstance(value, bool | int | float) or name == "n_support"
    }
    if scalars:
        st.dataframe(
            {"Propriedade": list(scalars), "Valor": list(scalars.values())},
            hide_index=True,
            width="stretch",
        )
        shown = True
    if not shown:
        st.info("Este modelo não mostra o que aprendeu.", icon=":material/info:")
    with st.expander("Estado completo, em JSON"):
        st.json(state, expanded=False)


def _coefficients(result: ExperimentResult) -> bool:
    """A tabela dos pesos de um modelo linear (se os pesos corresponderem às features)."""
    state = result.model_state
    coefficients: list[Any] | None = state.get("coefficients")
    if not coefficients:
        return False
    intercept = state.get("intercept", [])
    intercepts = intercept if isinstance(intercept, list) else [intercept]
    if not isinstance(coefficients[0], list):  # regressão: um só vetor de pesos
        rows: list[list[float]] = [coefficients]
        names = ["Peso"]
        meaning = "Cada peso é quanto a previsão muda quando a feature sobe uma unidade."
    else:
        rows = coefficients
        labels = result.class_labels
        classes = state.get("classes", list(range(len(labels))))
        # Na classificação binária há um só vetor de pesos: o da classe positiva.
        chosen = [labels[-1]] if len(rows) == 1 else [labels[int(code)] for code in classes]
        names = [f"Peso para «{label}»" for label in chosen]
        meaning = "Um peso positivo empurra para a classe, um negativo afasta."
    if any(len(row) != len(result.feature_names) for row in rows):
        st.caption(
            f"O modelo tem {len(rows[0])} pesos, um por termo (não por feature): por exemplo, as "
            "potências da regressão polinomial."
        )
        return False
    table: dict[str, Any] = {"Feature": list(result.feature_names)}
    for name, weights in zip(names, rows, strict=True):
        table[name] = [format_number(weight, 3) for weight in weights]
    st.dataframe(table, hide_index=True, width="stretch")
    written = ", ".join(format_number(value, 3) for value in intercepts)
    st.caption(
        f"Ordenada (b): {written}. {meaning} Com as features padronizadas, os pesos comparam-se "
        "entre si: os maiores (em valor absoluto) são as features que mais pesam na previsão."
    )
    return True


def _rows_table(result: ExperimentResult, key: str, title: str) -> bool:
    """Uma tabela com uma linha por grupo (ou componente) e uma coluna por feature."""
    rows = result.model_state.get(key)
    if not rows or len(rows[0]) != len(result.feature_names):
        return False
    if key == "components":
        names = [f"Componente {number}" for number in range(1, len(rows) + 1)]
    elif result.task_type.is_classification:
        classes = result.model_state.get("classes", list(range(len(rows))))
        names = [result.class_labels[int(code)] for code in classes]
    else:
        names = [f"Grupo {number}" for number in range(1, len(rows) + 1)]
    table: dict[str, Any] = {"": names}
    for column, feature in enumerate(result.feature_names):
        table[feature] = [format_number(row[column], 3) for row in rows]
    st.markdown(f"**{title}** (na escala depois da preparação)")
    st.dataframe(table, hide_index=True, width="stretch")
    return True


def _written(value: Any) -> str:
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_number(value, 4)
    if isinstance(value, list):
        return " + ".join(str(item) for item in value) + f" = {sum(value)}"
    return str(value)


main()
