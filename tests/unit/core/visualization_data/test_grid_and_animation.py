"""A grelha da sonda e a animação do treino, com dados preparados de verdade."""

from typing import Any

import numpy as np
import pytest

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.visualization_data.animation import MAX_POINTS, build_animation
from ml_lab.core.visualization_data.grid import (
    GRID_RESOLUTION,
    LINE_RESOLUTION,
    ProbeGrid,
    probe_grid,
)


def prepared(source: SourceSpec, target: str) -> Dataset:
    return DatasetService().prepared(source, PreparationConfig(target=target), 0).dataset


def synthetic(generator: str, **parameters: Any) -> SourceSpec:
    return SourceSpec("synthetic", {"generator": generator, "seed": 0, "parameters": parameters})


@pytest.fixture(scope="module")
def line() -> Dataset:
    return prepared(synthetic("regression", n_samples=60, n_features=1), "target")


@pytest.fixture(scope="module")
def moons() -> Dataset:
    return prepared(synthetic("moons", n_samples=2_000), "target")


@pytest.fixture(scope="module")
def iris() -> Dataset:
    return prepared(SourceSpec("sample", {"name": "iris"}), "species")


def grid_of(dataset: Dataset) -> ProbeGrid:
    grid = probe_grid(dataset)
    assert grid is not None
    return grid


def test_with_one_feature_the_grid_is_a_line(line: Dataset) -> None:
    grid = grid_of(line)

    assert grid.dimensions == 1
    assert grid.shape == (1, LINE_RESOLUTION)
    assert grid.points.shape == (LINE_RESOLUTION, 1)
    assert grid.axis_labels == ("x1", "target")
    train = line.train.X[:, 0]
    assert grid.xs[0] < train.min() < train.max() < grid.xs[-1]  # com margem dos dois lados


@pytest.mark.parametrize("name", ["moons", "iris"])
def test_the_plane_grid_goes_back_to_the_features_and_returns(
    name: str, request: pytest.FixtureRequest
) -> None:
    dataset: Dataset = request.getfixturevalue(name)
    grid = grid_of(dataset)
    projection = dataset.projection
    assert projection is not None

    plane = np.column_stack([coordinate.ravel() for coordinate in np.meshgrid(grid.xs, grid.ys)])

    assert grid.dimensions == 2
    assert grid.shape == (GRID_RESOLUTION, GRID_RESOLUTION)
    assert grid.points.shape == (GRID_RESOLUTION**2, dataset.n_features)
    assert np.allclose(projection.project(grid.points), plane)


def binary_snapshots(count: int) -> list[tuple[int, dict[str, Any]]]:
    """Snapshots falsos: a probabilidade da classe 1 cresce com o passo."""
    snapshots = []
    for step in range(1, count + 1):
        positive = np.full(GRID_RESOLUTION**2, step / (count + 1))
        snapshots.append((step, {"probe": np.column_stack([1 - positive, positive])}))
    return snapshots


def test_binary_frames_show_the_probability_of_the_positive_class(moons: Dataset) -> None:
    animation = build_animation(
        moons,
        grid_of(moons),
        binary_snapshots(3),
        losses={1: 0.6, 2: 0.5, 3: 0.4},
        classes=np.array([0, 1]),
    )

    assert animation is not None
    assert animation.animated
    assert animation.value_kind == "probability"
    assert animation.value_label == "Probabilidade de «1»"
    assert [frame.step for frame in animation.frames] == [1, 2, 3]
    assert [frame.loss for frame in animation.frames] == [0.6, 0.5, 0.4]
    assert animation.frames[0].values[0][0] == pytest.approx(0.25)
    assert len(animation.frames[0].values) == GRID_RESOLUTION
    assert moons.projection is not None
    assert animation.axis_labels == moons.projection.axis_labels


def test_multiclass_frames_show_the_predicted_class(iris: Dataset) -> None:
    probabilities = np.tile([0.1, 0.7, 0.2], (GRID_RESOLUTION**2, 1))

    animation = build_animation(
        iris, grid_of(iris), [(0, {"probe": probabilities})], classes=np.array([0, 1, 2])
    )

    assert animation is not None
    assert not animation.animated
    assert animation.value_kind == "class"
    assert {value for row in animation.frames[0].values for value in row} == {1.0}
    assert animation.class_labels == ("setosa", "versicolor", "virginica")


def test_regression_frames_show_the_predicted_value(line: Dataset) -> None:
    animation = build_animation(
        line, grid_of(line), [(5, {"probe": np.linspace(0, 1, LINE_RESOLUTION)})]
    )

    assert animation is not None
    assert animation.value_kind == "value"
    assert animation.value_label == "Previsão de target"
    assert animation.dimensions == 1
    assert len(animation.frames[0].values) == 1
    assert animation.points.values  # a nuvem de pontos traz o alvo verdadeiro
    assert len(animation.points.x) == len(line.train)


def test_without_predictions_on_the_probe_there_is_nothing_to_animate(moons: Dataset) -> None:
    assert build_animation(moons, grid_of(moons), [(1, {"weights": [0.1]})]) is None


def test_big_datasets_draw_a_sample_of_the_training_points(moons: Dataset) -> None:
    animation = build_animation(moons, grid_of(moons), binary_snapshots(1))

    assert animation is not None
    points = animation.points
    assert len(points.x) <= MAX_POINTS < len(moons.train)
    assert set(points.labels) == {"0", "1"}
    assert len(points.labels) == len(points.x) == len(points.y)
