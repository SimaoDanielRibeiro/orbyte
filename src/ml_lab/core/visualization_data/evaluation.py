"""Dados dos gráficos da avaliação, calculados no teste (só números; desenhar é da interface).

Cada tipo de tarefa tem os seus gráficos:

- **classificação**: as curvas ROC e de precisão-recall (uma por classe, em multiclasse);
- **regressão**: os resíduos (previsto contra erro) e a distribuição dos erros;
- **clustering**: o perfil de silhueta de cada grupo e, no hierárquico, o dendrograma;
- **redução de dimensionalidade**: a variância explicada por cada componente (na PCA);
- **anomalias**: a distribuição das pontuações e o limiar.

Em todos, os pontos de teste no plano da projeção 2D, pintados com o que interessa:
certo ou errado, o erro, o grupo ou a anomalia.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics as sk

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.evaluation.anomaly import FLAG_LABELS
from ml_lab.core.evaluation.clustering import NOISE_LABEL, group_name
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import LabModel, class_probabilities
from ml_lab.core.visualization_data.animation import MAX_POINTS, PointCloud

CURVE_POINTS: Final = 200
"""Pontos de cada curva ROC ou PR, no máximo."""

HISTOGRAM_BINS: Final = 30

SILHOUETTE_SAMPLE: Final = 2_000

DENDROGRAM_LEAVES: Final = 30
"""O dendrograma mostra só as últimas junções: os grupos mais pequenos ficam numa folha."""

CORRECT, WRONG = "Right", "Wrong"


@dataclass(frozen=True, slots=True)
class Curve:
    """Uma curva (ROC ou PR) de uma classe, com a área por baixo dela."""

    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    area: float


@dataclass(frozen=True, slots=True)
class Distribution:
    """Contagens por intervalo: ``edges`` tem mais um valor do que cada contagem."""

    edges: tuple[float, ...]
    counts: tuple[tuple[int, ...], ...]
    series: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Residuals:
    """O valor previsto, o real e o erro (real - previsto) de cada linha de teste."""

    predicted: tuple[float, ...]
    actual: tuple[float, ...]
    errors: tuple[float, ...]
    histogram: Distribution


@dataclass(frozen=True, slots=True)
class SilhouetteGroup:
    """Os valores de silhueta dos pontos de um grupo, do maior para o menor."""

    label: str
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Silhouette:
    """O perfil de silhueta: cada grupo e a média geral."""

    groups: tuple[SilhouetteGroup, ...]
    average: float


@dataclass(frozen=True, slots=True)
class Dendrogram:
    """O dendrograma, em segmentos prontos a desenhar.

    Attributes:
        segments: Cada junção é um "U" invertido: quatro pontos (x, altura).
        leaf_positions: A posição horizontal de cada folha.
        leaf_labels: Quantos pontos tem cada folha.
        cut_height: A altura onde o corte dá o número de grupos escolhido.
    """

    segments: tuple[tuple[tuple[float, float], ...], ...]
    leaf_positions: tuple[float, ...]
    leaf_labels: tuple[str, ...]
    cut_height: float | None


@dataclass(frozen=True, slots=True)
class EvaluationCharts:
    """Os dados dos gráficos de um run (os campos que não se aplicam à tarefa ficam vazios).

    Attributes:
        task_type: O tipo de tarefa.
        roc: As curvas ROC (classificação com probabilidades; anomalias com referência).
        pr: As curvas de precisão-recall.
        residuals: Na regressão, os resíduos.
        silhouette: No clustering, o perfil de silhueta.
        dendrogram: No clustering hierárquico, o dendrograma.
        cluster_sizes: No clustering, quantos pontos de teste tem cada grupo.
        explained_variance: Na PCA, a proporção da variância de cada componente.
        scores: Nas anomalias, a distribuição das pontuações (normal e anomalia).
        scores_reference: Se as pontuações estão separadas pelas anomalias verdadeiras (da
            coluna de referência) e não pelo que o detetor decidiu.
        threshold: Nas anomalias, o limiar a partir do qual um ponto é anomalia.
        test_points: Os pontos de teste no plano 2D, com o que interessa em ``labels``
            (ou ``values``).
        axis_labels: Os nomes dos eixos do plano.
    """

    task_type: TaskType
    roc: tuple[Curve, ...] = ()
    pr: tuple[Curve, ...] = ()
    residuals: Residuals | None = None
    silhouette: Silhouette | None = None
    dendrogram: Dendrogram | None = None
    cluster_sizes: tuple[tuple[str, int], ...] = ()
    explained_variance: tuple[float, ...] = ()
    scores: Distribution | None = None
    scores_reference: bool = False
    threshold: float | None = None
    test_points: PointCloud | None = None
    axis_labels: tuple[str, str] = ("", "")


def evaluation_charts(model: LabModel[Any], dataset: Dataset) -> EvaluationCharts:
    """Os dados dos gráficos da avaliação, no teste (no treino, se o modelo só souber esse)."""
    partition = Partition.TEST if model.info.supports_new_data else Partition.TRAIN
    task = dataset.task_type
    if task.is_classification:
        return _classification(model, dataset, partition)
    if task is TaskType.REGRESSION:
        return _regression(model, dataset, partition)
    if task is TaskType.CLUSTERING:
        return _clustering(model, dataset, partition)
    if task is TaskType.ANOMALY_DETECTION:
        return _anomalies(model, dataset, partition)
    ratios = tuple(float(value) for value in model.state().get("explained_variance_ratio", ()))
    return EvaluationCharts(task, explained_variance=ratios, axis_labels=_axes(dataset))


def _classification(
    model: LabModel[Any], dataset: Dataset, partition: Partition
) -> EvaluationCharts:
    split = dataset.splits[partition]
    y = _target(split.y).astype(np.int64)
    predicted = np.asarray(model.predict(split.X), dtype=np.int64)
    marks = [CORRECT if ok else WRONG for ok in predicted == y]
    roc: tuple[Curve, ...] = ()
    pr: tuple[Curve, ...] = ()
    if model.info.supports_proba:
        proba = class_probabilities(model, split.X, dataset.n_classes)
        roc, pr = classification_curves(y, proba, dataset)
    return EvaluationCharts(
        dataset.task_type,
        roc=roc,
        pr=pr,
        test_points=_plane(dataset, partition, labels=marks),
        axis_labels=_axes(dataset),
    )


def _regression(model: LabModel[Any], dataset: Dataset, partition: Partition) -> EvaluationCharts:
    split = dataset.splits[partition]
    actual = _target(split.y).astype(np.float64)
    predicted = np.asarray(model.predict(split.X), dtype=np.float64)
    return EvaluationCharts(
        dataset.task_type,
        residuals=residuals(actual, predicted),
        test_points=_plane(dataset, partition, values=np.abs(actual - predicted)),
        axis_labels=_axes(dataset),
    )


def _clustering(model: LabModel[Any], dataset: Dataset, partition: Partition) -> EvaluationCharts:
    split = dataset.splits[partition]
    labels = np.asarray(model.predict(split.X), dtype=np.int64)
    names = [group_name(group) for group in labels.tolist()]
    sizes = sorted({name: names.count(name) for name in names}.items(), key=_group_order)
    state = model.state()
    linkage = state.get("linkage")
    return EvaluationCharts(
        dataset.task_type,
        silhouette=silhouette_profile(split.X, labels),
        dendrogram=dendrogram(linkage, int(state["n_clusters"])) if linkage else None,
        cluster_sizes=tuple(sizes),
        test_points=_plane(dataset, partition, labels=names),
        axis_labels=_axes(dataset),
    )


def _anomalies(model: LabModel[Any], dataset: Dataset, partition: Partition) -> EvaluationCharts:
    split = dataset.splits[partition]
    flags = np.asarray(model.predict(split.X), dtype=np.int64)
    scores = np.asarray(model.anomaly_scores(split.X), dtype=np.float64)
    # Com uma referência de duas classes, a preparação deu o 1 à anomalia (a mais rara).
    truth = split.y.astype(np.int64) if split.y is not None and dataset.n_classes == 2 else None
    roc: tuple[Curve, ...] = ()
    if truth is not None and len(np.unique(truth)) == 2:
        roc = (_roc(truth, scores, "Anomaly"),)
    # Com referência, separar pelas anomalias verdadeiras mostra o que o limiar deixa escapar
    # (anomalias abaixo dele) e os falsos alarmes (normais acima); sem ela, só há a decisão.
    groups = flags if truth is None else truth
    return EvaluationCharts(
        dataset.task_type,
        roc=roc,
        scores=histogram(scores, [scores[groups == 0], scores[groups == 1]], FLAG_LABELS),
        scores_reference=truth is not None,
        threshold=model.state().get("threshold"),
        test_points=_plane(dataset, partition, labels=[FLAG_LABELS[flag] for flag in flags]),
        axis_labels=_axes(dataset),
    )


def _axes(dataset: Dataset) -> tuple[str, str]:
    return dataset.projection.axis_labels if dataset.projection else ("", "")


def _target(y: NDArray[Any] | None) -> NDArray[Any]:
    assert y is not None  # as tarefas supervisionadas têm sempre alvo
    return y


def classification_curves(
    y: NDArray[np.int64], proba: NDArray[np.float64], dataset: Dataset
) -> tuple[tuple[Curve, ...], tuple[Curve, ...]]:
    """As curvas ROC e PR: da classe positiva (binária) ou de cada classe contra as outras."""
    labels = dataset.class_labels
    classes = [1] if len(labels) == 2 else list(range(len(labels)))
    roc, pr = [], []
    for code in classes:
        truth = (y == code).astype(np.int64)
        if len(np.unique(truth)) < 2:  # a classe não aparece (ou só ela aparece) no teste
            continue
        roc.append(_roc(truth, proba[:, code], labels[code]))
        precision, recall, _ = sk.precision_recall_curve(truth, proba[:, code])
        area = float(sk.average_precision_score(truth, proba[:, code]))
        pr.append(Curve(labels[code], *_thin(recall, precision), area))
    return tuple(roc), tuple(pr)


def residuals(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> Residuals:
    """Os resíduos (real - previsto), numa amostra de até ``MAX_POINTS`` linhas."""
    errors = actual - predicted
    step = max(1, -(-len(errors) // MAX_POINTS))
    return Residuals(
        predicted=tuple(float(v) for v in predicted[::step]),
        actual=tuple(float(v) for v in actual[::step]),
        errors=tuple(float(v) for v in errors[::step]),
        histogram=histogram(errors, [errors], ("Errors",)),
    )


def histogram(
    everything: NDArray[np.float64], series: Sequence[NDArray[np.float64]], names: Sequence[str]
) -> Distribution:
    """As contagens de cada série nos mesmos intervalos (os de ``everything``)."""
    edges = np.histogram_bin_edges(everything, bins=HISTOGRAM_BINS)
    counts = tuple(tuple(int(c) for c in np.histogram(values, bins=edges)[0]) for values in series)
    return Distribution(tuple(float(e) for e in edges), counts, tuple(names))


def silhouette_profile(X: NDArray[np.float64], labels: NDArray[np.int64]) -> Silhouette | None:
    """A silhueta de cada ponto, por grupo (sem o ruído), numa amostra fixa de pontos."""
    kept = labels >= 0
    X, labels = X[kept], labels[kept]
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return None
    if len(X) > SILHOUETTE_SAMPLE:
        chosen = np.random.default_rng(0).choice(len(X), size=SILHOUETTE_SAMPLE, replace=False)
        X, labels = X[chosen], labels[chosen]
    values = sk.silhouette_samples(X, labels)
    groups = tuple(
        SilhouetteGroup(
            group_name(int(group)), tuple(float(v) for v in np.sort(values[labels == group])[::-1])
        )
        for group in np.unique(labels)
    )
    return Silhouette(groups, float(values.mean()))


def dendrogram(
    linkage: Sequence[Sequence[float]], n_clusters: int, leaves: int = DENDROGRAM_LEAVES
) -> Dendrogram:
    """O dendrograma das últimas ``leaves - 1`` junções (como o ``truncate_mode="lastp"`` do SciPy).

    Os índices até ``n - 1`` são pontos; ``n + i`` é o grupo criado na junção ``i``. Os
    grupos criados antes das últimas junções ficam como folhas, com o número de pontos.
    """
    merges = np.asarray(linkage, dtype=np.float64)
    n_points = len(merges) + 1
    shown = min(leaves, n_points)
    first_shown = 2 * n_points - shown  # os nós a partir daqui desenham-se como junções
    positions: dict[int, float] = {}
    heights: dict[int, float] = {}
    leaf_positions: list[float] = []
    leaf_labels: list[str] = []
    segments: list[tuple[tuple[float, float], ...]] = []

    def size(node: int) -> int:
        return 1 if node < n_points else int(merges[node - n_points, 3])

    def place(node: int) -> None:
        if node < first_shown:  # folha: um ponto, ou um grupo encolhido
            positions[node] = 10.0 * len(leaf_positions) + 5.0
            heights[node] = 0.0
            leaf_positions.append(positions[node])
            count = size(node)
            leaf_labels.append("1" if count == 1 else f"({count})")
            return
        left, right, height, _ = merges[node - n_points]
        place(int(left))
        place(int(right))
        a, b = int(left), int(right)
        positions[node] = (positions[a] + positions[b]) / 2
        heights[node] = float(height)
        segments.append(
            (
                (positions[a], heights[a]),
                (positions[a], float(height)),
                (positions[b], float(height)),
                (positions[b], heights[b]),
            )
        )

    place(2 * n_points - 2)
    cut = None
    if 2 <= n_clusters < n_points:
        below, above = merges[n_points - n_clusters - 1, 2], merges[n_points - n_clusters, 2]
        cut = float((below + above) / 2)
    return Dendrogram(tuple(segments), tuple(leaf_positions), tuple(leaf_labels), cut)


def _roc(truth: NDArray[np.int64], scores: NDArray[np.float64], label: str) -> Curve:
    false_positive, true_positive, _ = sk.roc_curve(truth, scores)
    area = float(sk.roc_auc_score(truth, scores))
    return Curve(label, *_thin(false_positive, true_positive), area)


def _thin(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """No máximo ``CURVE_POINTS`` pontos, mantendo o primeiro e o último."""
    if len(x) > CURVE_POINTS:
        chosen = np.unique(np.linspace(0, len(x) - 1, CURVE_POINTS).astype(int))
        x, y = x[chosen], y[chosen]
    return tuple(float(v) for v in x), tuple(float(v) for v in y)


def _plane(
    dataset: Dataset,
    partition: Partition,
    *,
    labels: Sequence[str] = (),
    values: NDArray[np.float64] | None = None,
) -> PointCloud | None:
    """Os pontos de uma parte no plano 2D, com uma etiqueta (ou um valor) cada."""
    if dataset.projection is None:
        return None
    plane = dataset.projection.coordinates[partition]
    step = max(1, -(-len(plane) // MAX_POINTS))
    return PointCloud(
        x=tuple(float(v) for v in plane[::step, 0]),
        y=tuple(float(v) for v in plane[::step, 1]),
        labels=tuple(labels[::step]),
        values=() if values is None else tuple(float(v) for v in values[::step]),
    )


def _group_order(item: tuple[str, int]) -> tuple[int, int]:
    name = item[0]
    return (1, 0) if name == NOISE_LABEL else (0, int(name.split()[-1]))
