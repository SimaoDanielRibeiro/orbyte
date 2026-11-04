"""Animação do treino: o campo do modelo (fronteira, grupos, previsão) em cada snapshot.

Cada snapshot traz as previsões do modelo nos pontos da sonda. Aqui essas previsões
viram "campos" prontos a desenhar, um por snapshot: a probabilidade da classe positiva
(classificação binária), a classe ou o grupo previsto (multiclasse, clustering), a
pontuação de anomalia ou o valor previsto (regressão). Se o estado do modelo tiver
centros (centróides, médias de cada classe) e covariâncias (as nuvens de uma mistura
gaussiana), vão também para o plano, como pontos e elipses. Junta-se a nuvem dos pontos
de treino. Só números: desenhar é com a interface (regra 2 da interface, em
``docs/architecture.md``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.projection.projection_2d import Projection
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.visualization_data.grid import ProbeGrid

type ValueKind = Literal["probability", "class", "value", "score"]
"""O que os valores do campo são: probabilidade, classe (ou grupo), valor previsto ou
pontuação de anomalia."""

MAX_POINTS: Final = 800
"""Pontos de treino desenhados por baixo da fronteira, no máximo."""

ELLIPSE_POINTS: Final = 60
"""Pontos do contorno de cada elipse."""

ELLIPSE_SIGMAS: Final = 2.0
"""As elipses mostram a zona a dois desvios padrão do centro (cerca de 86% da nuvem, em 2D)."""

NOISE_LABEL: Final = "Noise"


@dataclass(frozen=True, slots=True)
class Outline:
    """Um contorno fechado no plano (por exemplo, uma elipse), ponto a ponto."""

    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FieldFrame:
    """O campo num passo do treino.

    Attributes:
        step: O passo (0 para o modelo final de um modelo que treina de uma vez).
        values: Os valores da grelha, por linhas (em 1D, uma só linha).
        loss: A loss nesse passo, se o modelo a anunciou.
        centers: Os centros do modelo no plano (centróides, médias), se os tiver.
        outlines: As elipses das nuvens do modelo, se as tiver.
    """

    step: int
    values: tuple[tuple[float, ...], ...]
    loss: float | None = None
    centers: tuple[tuple[float, float], ...] = ()
    outlines: tuple[Outline, ...] = ()


@dataclass(frozen=True, slots=True)
class PointCloud:
    """Os pontos de treino, no mesmo plano (ou eixo) da grelha.

    Attributes:
        x: Coordenada horizontal.
        y: Coordenada vertical (em 1D, o alvo).
        labels: A classe de cada ponto (do alvo, ou da coluna de referência).
        values: Na regressão, o valor do alvo de cada ponto.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    labels: tuple[str, ...] = ()
    values: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingAnimation:
    """Tudo o que é preciso para animar o treino.

    Attributes:
        dimensions: 2 (campo no plano) ou 1 (curva ao longo de uma feature).
        value_kind: O que os valores do campo são.
        value_label: O nome a mostrar para os valores.
        xs: A grelha no eixo horizontal.
        ys: A grelha no eixo vertical (vazio em 1D).
        axis_labels: O nome de cada eixo.
        class_labels: O nome das classes dos pontos (do alvo ou da referência).
        frames: Um campo por snapshot, por ordem de passo.
        points: Os pontos de treino.
        field_labels: Quando o campo são categorias, o nome de cada valor (o valor ``k`` é
            ``field_labels[k]``): as classes, ou os grupos do clustering.
        noise: Se a última categoria do campo é o ruído (pontos sem grupo).
        how_to_read: Como se lê o desenho, numa ou duas frases (a interface mostra-o por
            baixo dele).
    """

    dimensions: int
    value_kind: ValueKind
    value_label: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    axis_labels: tuple[str, str]
    class_labels: tuple[str, ...]
    frames: tuple[FieldFrame, ...]
    points: PointCloud
    field_labels: tuple[str, ...] = ()
    noise: bool = False
    how_to_read: str = ""

    @property
    def animated(self) -> bool:
        """Se há mais do que uma fotografia (um modelo que treinou passo a passo)."""
        return len(self.frames) > 1


def build_animation(
    dataset: Dataset,
    grid: ProbeGrid,
    snapshots: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    losses: Mapping[int, float | None] | None = None,
    classes: NDArray[np.int64] | None = None,
) -> TrainingAnimation | None:
    """Monta a animação a partir dos snapshots que têm previsões na sonda.

    Args:
        dataset: Os dados (para a nuvem de pontos e os nomes das classes).
        grid: A grelha onde se pediram as previsões.
        snapshots: Pares ``(passo, estado)``; o estado traz as previsões em ``"probe"``.
        losses: A loss de cada passo, para mostrar junto de cada fotografia.
        classes: Na classificação, as classes que o modelo viu no treino (as colunas das
            probabilidades).

    Returns:
        A animação, ou ``None`` se nenhum snapshot tiver previsões na sonda.
    """
    fields: list[tuple[int, NDArray[np.float64], Mapping[str, Any]]] = []
    kind: ValueKind = "value"
    for step, state in snapshots:
        probe = state.get("probe")
        if probe is None:
            continue
        field, kind = _field(np.asarray(probe, dtype=np.float64), dataset, classes)
        fields.append((step, field, state))
    if not fields:
        return None
    field_labels, noise = _categories(kind, dataset, [field for _, field, _ in fields])
    projection = dataset.projection if grid.dimensions == 2 else None
    frames = []
    for step, field, state in fields:
        if noise:  # o ruído (-1) passa a ser a última categoria
            field = np.where(field < 0, len(field_labels) - 1, field)
        frames.append(
            FieldFrame(
                step=step,
                values=tuple(tuple(float(v) for v in row) for row in field.reshape(grid.shape)),
                loss=(losses or {}).get(step),
                centers=_centers(state, projection),
                outlines=_outlines(state, projection),
            )
        )
    return TrainingAnimation(
        dimensions=grid.dimensions,
        value_kind=kind,
        value_label=_value_label(kind, dataset),
        xs=tuple(float(x) for x in grid.xs),
        ys=tuple(float(y) for y in grid.ys),
        axis_labels=grid.axis_labels,
        class_labels=dataset.class_labels,
        frames=tuple(frames),
        points=_points(dataset, grid),
        field_labels=field_labels,
        noise=noise,
        how_to_read=_how_to_read(kind, grid.dimensions, dataset.task_type, noise),
    )


def _how_to_read(kind: ValueKind, dimensions: int, task: TaskType, noise: bool) -> str:
    """Como se lê o campo, conforme o que os valores são."""
    if kind == "probability" and dimensions == 2:
        return (
            "The colours are the probability the model gives each class at each point of the "
            "plane; the line is the decision boundary (50%). The points are the training data."
        )
    if kind == "probability":
        return (
            "The curve is the probability of the positive class along the feature; the "
            "decision boundary is where the curve crosses 50%. The points are the training "
            "data."
        )
    if kind == "class" and task is TaskType.CLUSTERING:
        text = (
            "Each colour is the group the model would give a point in that area of the plane; "
            "the crosses are the centres and the dotted lines the ellipses of the clouds, if "
            "the model has them. The points are the training data, shaped by the reference "
            "column (if you picked one)."
        )
        return f"{text} The grey areas are noise (they belong to no group)." if noise else text
    if kind == "class":
        return (
            "Each colour is the class the model predicts in that area of the plane: the "
            "decision boundaries are where the colour changes. The points are the training "
            "data."
        )
    if kind == "score":
        return (
            "The colours are the anomaly score at each point of the plane: the stronger the "
            "colour, the stranger a point there would be. The points are the training data."
        )
    if dimensions == 1:
        return "The line is the model's prediction; the points are the training data."
    return (
        "The colours are the value the model predicts at each point of the plane; the points "
        "are the training data, coloured by the real value on the same scale."
    )


def _field(
    probe: NDArray[np.float64], dataset: Dataset, classes: NDArray[np.int64] | None
) -> tuple[NDArray[np.float64], ValueKind]:
    """Os valores a desenhar em cada ponto da sonda, e o que significam."""
    task = dataset.task_type
    if task is TaskType.CLUSTERING:
        groups = (
            np.argmax(probe, axis=1) if probe.ndim == 2 else probe
        )  # a mistura dá probabilidades
        return groups.astype(np.float64), "class"
    if task is TaskType.ANOMALY_DETECTION:
        return probe.reshape(-1), "score"
    if not task.is_classification:
        return probe.reshape(-1), "value"
    if probe.ndim == 1:  # o modelo não dá probabilidades: só a classe prevista
        return probe, "class"
    seen = classes if classes is not None else np.arange(probe.shape[1])
    if dataset.n_classes == 2:
        positive = np.flatnonzero(seen == 1)
        if positive.size:
            return probe[:, int(positive[0])], "probability"
        return np.zeros(len(probe)), "probability"
    return seen[np.argmax(probe, axis=1)].astype(np.float64), "class"


def _categories(
    kind: ValueKind, dataset: Dataset, fields: Sequence[NDArray[np.float64]]
) -> tuple[tuple[str, ...], bool]:
    """O nome de cada valor do campo, quando são categorias, e se há ruído."""
    if kind != "class":
        return (), False
    if dataset.task_type is not TaskType.CLUSTERING:
        return dataset.class_labels, False
    largest = max(int(field.max()) for field in fields)
    names = tuple(f"Group {group + 1}" for group in range(largest + 1))
    noise = any(bool((field < 0).any()) for field in fields)
    return ((*names, NOISE_LABEL) if noise else names), noise


def _value_label(kind: ValueKind, dataset: Dataset) -> str:
    task = dataset.task_type
    if kind == "probability":
        return f"Probability of “{dataset.class_labels[1]}”"
    if task is TaskType.CLUSTERING:
        return "Predicted group"
    if kind == "class":
        return "Predicted class"
    if task is TaskType.ANOMALY_DETECTION:
        return "Anomaly score"
    return f"Prediction of {dataset.target_name}" if dataset.target_name else "Prediction"


def _centers(
    state: Mapping[str, Any], projection: Projection | None
) -> tuple[tuple[float, float], ...]:
    """Os centros do modelo (``state["centers"]``, no espaço das features) levados ao plano."""
    centers = state.get("centers")
    if projection is None or not centers:
        return ()
    plane = projection.project(np.asarray(centers, dtype=np.float64))
    return tuple((float(x), float(y)) for x, y in plane)


def _outlines(state: Mapping[str, Any], projection: Projection | None) -> tuple[Outline, ...]:
    """As elipses das nuvens (``state["centers"]`` e ``state["covariances"]``) no plano."""
    centers, covariances = state.get("centers"), state.get("covariances")
    if projection is None or not centers or not covariances:
        return ()
    plane = projection.project(np.asarray(centers, dtype=np.float64))
    angles = np.linspace(0, 2 * np.pi, ELLIPSE_POINTS)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    outlines = []
    for center, covariance in zip(plane, covariances, strict=True):
        shape = projection.project_covariance(np.asarray(covariance, dtype=np.float64))
        values, vectors = np.linalg.eigh(shape)
        # Cada direção própria esticada pelo desvio padrão nessa direção.
        axes = vectors * (ELLIPSE_SIGMAS * np.sqrt(np.maximum(values, 0.0)))
        contour = center[:, None] + axes @ circle
        outlines.append(
            Outline(tuple(float(v) for v in contour[0]), tuple(float(v) for v in contour[1]))
        )
    return tuple(outlines)


def _points(dataset: Dataset, grid: ProbeGrid) -> PointCloud:
    train = dataset.train
    step = max(1, -(-len(train) // MAX_POINTS))
    target = train.y[::step] if train.y is not None else None
    if grid.dimensions == 1:
        x = train.X[::step, 0]
        y = target if target is not None else np.zeros(len(x))
    else:
        projection = dataset.projection
        assert projection is not None  # a grelha 2D só existe quando há projeção
        plane = projection.coordinates[Partition.TRAIN][::step]
        x, y = plane[:, 0], plane[:, 1]
    labels: tuple[str, ...] = ()
    values: tuple[float, ...] = ()
    if target is not None and dataset.class_labels:
        labels = tuple(dataset.class_labels[int(code)] for code in target)
    elif target is not None:
        values = tuple(float(v) for v in target)
    return PointCloud(
        x=tuple(float(v) for v in x), y=tuple(float(v) for v in y), labels=labels, values=values
    )
