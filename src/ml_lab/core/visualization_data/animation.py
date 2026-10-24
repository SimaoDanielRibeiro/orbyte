"""Animação do treino: a fronteira de decisão (ou a reta da regressão) em cada snapshot.

Cada snapshot traz as previsões do modelo nos pontos da sonda. Aqui essas previsões
viram "campos" prontos a desenhar, um por snapshot: a probabilidade da classe positiva
(classificação binária), a classe prevista (multiclasse) ou o valor previsto
(regressão). Junta-se a nuvem dos pontos de treino, para se ver a fronteira a
separá-los. Só números: desenhar é com a interface (regra das duas interfaces).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.foundation.types import Partition
from ml_lab.core.visualization_data.grid import ProbeGrid

type ValueKind = Literal["probability", "class", "value"]

MAX_POINTS: Final = 800
"""Pontos de treino desenhados por baixo da fronteira, no máximo."""


@dataclass(frozen=True, slots=True)
class FieldFrame:
    """O campo num passo do treino.

    Attributes:
        step: O passo (0 para o modelo final de um modelo que treina de uma vez).
        values: Os valores da grelha, por linhas (em 1D, uma só linha).
        loss: A loss nesse passo, se o modelo a anunciou.
    """

    step: int
    values: tuple[tuple[float, ...], ...]
    loss: float | None = None


@dataclass(frozen=True, slots=True)
class PointCloud:
    """Os pontos de treino, no mesmo plano (ou eixo) da grelha.

    Attributes:
        x: Coordenada horizontal.
        y: Coordenada vertical (em 1D, o alvo).
        labels: Na classificação, a classe de cada ponto.
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
        dimensions: 2 (fronteira no plano) ou 1 (curva ao longo de uma feature).
        value_kind: O que os valores do campo são.
        value_label: O nome a mostrar para os valores.
        xs: A grelha no eixo horizontal.
        ys: A grelha no eixo vertical (vazio em 1D).
        axis_labels: O nome de cada eixo.
        class_labels: Na classificação, o nome de cada classe.
        frames: Um campo por snapshot, por ordem de passo.
        points: Os pontos de treino.
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
    frames: list[FieldFrame] = []
    kind: ValueKind = "value"
    for step, state in snapshots:
        probe = state.get("probe")
        if probe is None:
            continue
        field, kind = _field(np.asarray(probe, dtype=np.float64), dataset, classes)
        rows = field.reshape(grid.shape)
        frames.append(
            FieldFrame(
                step=step,
                values=tuple(tuple(float(v) for v in row) for row in rows),
                loss=(losses or {}).get(step),
            )
        )
    if not frames:
        return None
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
    )


def _field(
    probe: NDArray[np.float64], dataset: Dataset, classes: NDArray[np.int64] | None
) -> tuple[NDArray[np.float64], ValueKind]:
    """Os valores a desenhar em cada ponto da sonda, e o que significam."""
    if not dataset.task_type.is_classification:
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


def _value_label(kind: ValueKind, dataset: Dataset) -> str:
    if kind == "probability":
        return f"Probabilidade de «{dataset.class_labels[1]}»"
    if kind == "class":
        return "Classe prevista"
    return f"Previsão de {dataset.target_name}" if dataset.target_name else "Previsão"


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
