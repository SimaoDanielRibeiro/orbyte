"""Contrato do pacote: o modelo de cada um dos 34 vai para o pacote e volta a prever igual.

O ``model.joblib`` de um pacote é o modelo treinado, inteiro: depois de passar pelo ZIP, pela
verificação das impressões digitais e de voltar a abrir, tem de dar o mesmo (as previsões, ou a
projeção) que dava antes. Um modelo novo entra aqui sozinho.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.tracking.packages import (
    FORMAT,
    FORMAT_VERSION,
    INPUT,
    MODEL,
    PREPARATION,
    README,
    RUN,
    PackageFile,
    PackageKey,
    assemble,
    load_model,
    model_bytes,
    read_package,
)

type Data = tuple[NDArray[np.float64], NDArray[Any] | None, NDArray[np.float64]]

MANIFEST = {"format": FORMAT, "format_version": FORMAT_VERSION, "name": "Contrato"}

CASES = [
    pytest.param(name, task_type, id=f"{name}-{task_type}")
    for name, model_class in MODELS.items()
    for task_type in sorted(model_class.info.task_types)
]


def tiny_dataset(task_type: TaskType) -> Data:
    """Features de treino, o alvo e features novas."""
    rng = np.random.default_rng(5)
    if task_type.is_classification:
        n_classes = 2 if task_type is TaskType.BINARY_CLASSIFICATION else 3
        centers = rng.uniform(-4, 4, size=(n_classes, 3))
        y = np.repeat(np.arange(n_classes, dtype=np.int64), 20)
        X = centers[y] + rng.normal(size=(len(y), 3))
        return X, y, X[::4] + rng.normal(scale=0.3, size=X[::4].shape)
    if task_type is TaskType.REGRESSION:
        X = rng.normal(size=(60, 3))
        return X, X @ np.array([1.5, -2.0, 0.5]), rng.normal(size=(10, 3))
    centers = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    X = centers[np.repeat(np.arange(3), 20)] + rng.normal(scale=0.8, size=(60, 3))
    return X, None, X[::4] + rng.normal(scale=0.3, size=X[::4].shape)


def output(model: LabModel[Any], X: NDArray[np.float64]) -> NDArray[Any]:
    """A projeção, na redução de dimensionalidade; senão, as previsões."""
    if model.info.task_types == {TaskType.DIMENSIONALITY_REDUCTION}:
        return model.transform(X)
    return model.predict(X)


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_the_model_comes_back_from_the_package_and_predicts_the_same(
    name: str, task_type: TaskType, tmp_path: Path
) -> None:
    X, y, new = tiny_dataset(task_type)
    model = MODELS.get(name)(seed=1).fit(X, y)
    key = PackageKey(tmp_path / "package.key")
    parts = [
        PackageFile(README, b"# Contrato\n"),
        PackageFile(RUN, b"{}"),
        PackageFile(INPUT, b'{"columns": []}'),
        PackageFile(PREPARATION, b'{"transforms": []}'),
        PackageFile(MODEL, model_bytes(model)),
    ]

    opened = read_package(assemble(parts, MANIFEST, key).to_zip(), key)
    back = load_model(opened)

    # O t-SNE não coloca pontos novos: só se compara nos pontos do treino.
    points = new if model.info.supports_new_data else X
    assert opened.intact
    assert type(back) is type(model)
    np.testing.assert_array_equal(output(back, points), output(model, points))
