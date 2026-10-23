import numpy as np
import pytest

from ml_lab.core.data.projection.pca import fit_pca
from ml_lab.core.data.projection.projection_2d import ProjectionConfig, project_2d
from ml_lab.core.foundation.errors import (
    ColumnNotFoundError,
    DataError,
    InvalidParameterError,
)
from ml_lab.core.foundation.types import Partition


def stretched_cloud(n: int = 200) -> np.ndarray:
    """Pontos alongados ao longo da diagonal: a 1.ª componente tem de ser (1, 1)/√2."""
    rng = np.random.default_rng(0)
    t = rng.normal(scale=3.0, size=n)
    noise = rng.normal(scale=0.1, size=n)
    return np.column_stack([t + noise, t - noise, rng.normal(scale=0.5, size=n)])


def test_the_first_component_follows_the_largest_variance() -> None:
    pca = fit_pca(stretched_cloud(), n_components=2)

    assert np.allclose(np.abs(pca.components[0, :2]), 1 / np.sqrt(2), atol=0.01)
    assert pca.explained_variance_ratio[0] > 0.9
    assert pca.explained_variance_ratio.sum() <= 1.0


def test_components_are_orthonormal_and_signs_are_fixed() -> None:
    pca = fit_pca(stretched_cloud(), n_components=3)

    assert np.allclose(pca.components @ pca.components.T, np.eye(3))
    largest = np.abs(pca.components).argmax(axis=1)
    assert (pca.components[np.arange(3), largest] > 0).all()


def test_projected_training_data_is_centered() -> None:
    values = stretched_cloud()
    pca = fit_pca(values)

    coordinates = pca.transform(values)

    assert coordinates.shape == (200, 2)
    assert np.allclose(coordinates.mean(axis=0), 0.0)


def test_pca_needs_at_least_two_rows() -> None:
    with pytest.raises(DataError):
        fit_pca(np.ones((1, 3)))


def test_two_features_are_shown_as_they_are() -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0]])

    projection, step = project_2d({Partition.TRAIN: train}, ("a", "b"), ProjectionConfig())

    assert projection is not None
    assert step is not None
    assert projection.method == "columns"
    assert projection.axis_labels == ("a", "b")
    assert np.array_equal(projection.coordinates[Partition.TRAIN], train)


def test_more_features_use_pca_learned_on_the_training_data() -> None:
    values = stretched_cloud()
    matrices = {Partition.TRAIN: values[:150], Partition.TEST: values[150:]}

    projection, step = project_2d(matrices, ("x", "y", "z"), ProjectionConfig())

    assert projection is not None
    assert step is not None
    assert projection.method == "pca"
    assert projection.coordinates[Partition.TEST].shape == (50, 2)
    assert projection.loadings is not None
    assert "PC1 explica" in step.description


def test_columns_can_be_chosen() -> None:
    values = stretched_cloud()
    config = ProjectionConfig("columns", ("z", "x"))

    projection, _ = project_2d({Partition.TRAIN: values}, ("x", "y", "z"), config)

    assert projection is not None
    assert np.array_equal(projection.coordinates[Partition.TRAIN], values[:, [2, 0]])
    with pytest.raises(ColumnNotFoundError):
        project_2d(
            {Partition.TRAIN: values}, ("x", "y", "w"), ProjectionConfig("columns", ("z", "x"))
        )


def test_one_feature_has_no_projection_and_bad_configs_are_rejected() -> None:
    assert project_2d({Partition.TRAIN: np.ones((3, 1))}, ("x",), ProjectionConfig()) == (
        None,
        None,
    )
    with pytest.raises(InvalidParameterError):
        ProjectionConfig("columns")
    with pytest.raises(InvalidParameterError):
        ProjectionConfig("tsne")  # type: ignore[arg-type]
