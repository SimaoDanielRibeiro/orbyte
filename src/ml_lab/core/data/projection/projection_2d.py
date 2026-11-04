"""Versão 2D dos dados para os gráficos: duas colunas escolhidas ou as duas componentes da PCA.

Um gráfico só mostra duas dimensões. Com duas features, usam-se as próprias; com
mais, a PCA (aprendida só no treino) encontra o plano que mostra mais variância. É
uma cópia à parte: o modelo continua a treinar com todas as features.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.lineage import LineageStep, Stage
from ml_lab.core.data.projection.pca import fit_pca
from ml_lab.core.foundation.errors import ColumnNotFoundError, InvalidParameterError
from ml_lab.core.foundation.types import Partition
from ml_lab.formatting import format_percent

type ProjectionMethod = Literal["auto", "pca", "columns"]


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    """Como obter as duas dimensões.

    Attributes:
        method: ``"columns"`` (duas features escolhidas), ``"pca"`` ou ``"auto"`` (as
            próprias features se forem duas, PCA se forem mais).
        columns: As duas features, com ``method="columns"``.
    """

    method: ProjectionMethod = "auto"
    columns: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if self.method not in ("auto", "pca", "columns"):
            raise InvalidParameterError(f"Unknown projection method '{self.method}'.")
        if self.columns is not None:
            object.__setattr__(self, "columns", tuple(self.columns))
        if self.method == "columns" and (self.columns is None or len(self.columns) != 2):
            raise InvalidParameterError("The projection by columns needs exactly two columns.")


_NO_GEOMETRY = "This projection didn't keep its geometry."


@dataclass(frozen=True, slots=True, eq=False)
class Projection:
    """As coordenadas 2D de cada parte dos dados, e a forma de ir e voltar do plano.

    Attributes:
        method: ``"pca"`` ou ``"columns"``.
        axis_labels: Nome de cada eixo.
        coordinates: Para cada parte, uma matriz ``(n_linhas, 2)``.
        explained_variance_ratio: Na PCA, a proporção da variância explicada por cada eixo.
        loadings: Na PCA, o peso de cada feature em cada eixo, ``(2, n_features)``.
        feature_names: As features a que os pesos se referem.
        center: A média de cada feature no treino: o ponto do espaço das features que fica
            na origem do plano (na PCA) ou que dá o valor das features fora do plano.
        column_positions: Na projeção por colunas, a posição das duas features nos dados.
    """

    method: Literal["pca", "columns"]
    axis_labels: tuple[str, str]
    coordinates: Mapping[Partition, NDArray[np.float64]]
    explained_variance_ratio: tuple[float, float] | None = None
    loadings: NDArray[np.float64] | None = None
    feature_names: tuple[str, ...] = ()
    center: NDArray[np.float64] | None = None
    column_positions: tuple[int, int] | None = None

    def project(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Leva pontos do espaço das features para o plano (por exemplo, centróides)."""
        if self.method == "columns" and self.column_positions is not None:
            result: NDArray[np.float64] = points[:, list(self.column_positions)]
            return result
        if self.loadings is None or self.center is None:
            raise InvalidParameterError(_NO_GEOMETRY)
        projected: NDArray[np.float64] = (points - self.center) @ self.loadings.T
        return projected

    def to_features(self, plane: NDArray[np.float64]) -> NDArray[np.float64]:
        """Leva pontos do plano para o espaço das features.

        Na PCA, o ponto ``(a, b)`` do plano é ``centro + a·PC1 + b·PC2``; na projeção por
        colunas, é o centro com as duas features escolhidas trocadas por ``a`` e ``b``. As
        outras direções ficam na média do treino: o plano é um "corte" pelo meio dos dados.
        """
        if self.center is None:
            raise InvalidParameterError(_NO_GEOMETRY)
        if self.method == "columns" and self.column_positions is not None:
            points = np.tile(self.center, (len(plane), 1))
            points[:, list(self.column_positions)] = plane
            return points
        if self.loadings is None:
            raise InvalidParameterError(_NO_GEOMETRY)
        mapped: NDArray[np.float64] = self.center + plane @ self.loadings
        return mapped

    def project_covariance(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """A forma de uma nuvem (a sua matriz de covariância) vista no plano, ``(2, 2)``.

        Na PCA é ``L Σ Lᵀ`` (``L`` são os pesos dos dois eixos); na projeção por colunas, é a
        parte da matriz que diz respeito às duas features escolhidas.
        """
        if self.method == "columns" and self.column_positions is not None:
            positions = list(self.column_positions)
            selected: NDArray[np.float64] = covariance[np.ix_(positions, positions)]
            return selected
        if self.loadings is None:
            raise InvalidParameterError(_NO_GEOMETRY)
        projected: NDArray[np.float64] = self.loadings @ covariance @ self.loadings.T
        return projected


def project_2d(
    matrices: Mapping[Partition, NDArray[np.float64]],
    feature_names: tuple[str, ...],
    config: ProjectionConfig,
) -> tuple[Projection | None, LineageStep | None]:
    """Calcula a projeção 2D de cada parte.

    Args:
        matrices: As features de cada parte, já transformadas, com o treino incluído.
        feature_names: Os nomes das colunas das matrizes.
        config: O método.

    Returns:
        A projeção e o passo de linhagem, ou ``(None, None)`` se houver menos de duas
        features.
    """
    if len(feature_names) < 2:
        return None, None
    train = matrices[Partition.TRAIN]
    method = config.method
    if method == "auto":
        method = "columns" if len(feature_names) == 2 else "pca"

    if method == "columns":
        first, second = config.columns if config.columns is not None else feature_names[:2]
        positions = []
        for name in (first, second):
            if name not in feature_names:
                raise ColumnNotFoundError(name, feature_names)
            positions.append(feature_names.index(name))
        coordinates = {part: values[:, positions] for part, values in matrices.items()}
        projection = Projection(
            "columns",
            (first, second),
            coordinates,
            feature_names=feature_names,
            center=train.mean(axis=0),
            column_positions=(positions[0], positions[1]),
        )
        description = f"2D projection on the columns {first} and {second}."
    else:
        pca = fit_pca(train, n_components=2)
        ratios = tuple(float(r) for r in pca.explained_variance_ratio)
        projection = Projection(
            method="pca",
            axis_labels=("PC1", "PC2"),
            coordinates={part: pca.transform(values) for part, values in matrices.items()},
            explained_variance_ratio=(ratios[0], ratios[1]),
            loadings=pca.components,
            feature_names=feature_names,
            center=pca.mean,
        )
        description = (
            f"2D projection with PCA learned on the training data: PC1 explains "
            f"{format_percent(ratios[0])} and PC2 {format_percent(ratios[1])} of the variance of "
            f"the {len(feature_names)} features."
        )
    step = LineageStep(
        stage=Stage.PROJECTION,
        operation=f"projection_{projection.method}",
        description=description,
        params={"method": projection.method, "axes": list(projection.axis_labels)},
    )
    return projection, step
