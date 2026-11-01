"""Contrato do ONNX: cada modelo que sabe prever linhas novas passa para ONNX e prevê o mesmo.

O ONNX Runtime corre o ficheiro com as mesmas features, e o que sai tem de ser o que o
laboratório dá: a mesma classe, o mesmo grupo, a mesma marca de anomalia, os mesmos números (a
menos da precisão do float32) e as mesmas probabilidades. Um modelo novo entra aqui sozinho; se
não tiver conversão, o teste diz qual.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_lab.core.export import model_onnx, run_onnx
from ml_lab.core.foundation.errors import CapabilityError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, LabModel

type Data = tuple[NDArray[np.float64], NDArray[Any] | None, NDArray[np.float64]]

RELATIVE = 1e-4


def tiny_dataset(task_type: TaskType) -> Data:
    """Features de treino, o alvo e features novas (perto dos dados, mas não iguais)."""
    rng = np.random.default_rng(3)
    if task_type.is_classification:
        n_classes = 2 if task_type is TaskType.BINARY_CLASSIFICATION else 3
        centers = rng.uniform(-4, 4, size=(n_classes, 3))
        y = np.repeat(np.arange(n_classes, dtype=np.int64), 20)
        X = centers[y] + rng.normal(size=(len(y), 3))
        return X, y, X[::3] + rng.normal(scale=0.3, size=X[::3].shape)
    if task_type is TaskType.REGRESSION:
        X = rng.normal(size=(60, 3))
        target = X @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.1, size=60)
        return X, target, rng.normal(size=(12, 3))
    centers = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    X = centers[np.repeat(np.arange(3), 20)] + rng.normal(scale=0.8, size=(60, 3))
    return X, None, X[::4] + rng.normal(scale=0.3, size=X[::4].shape)


CASES = [
    pytest.param(name, task_type, id=f"{name}-{task_type}")
    for name, model_class in MODELS.items()
    for task_type in sorted(model_class.info.task_types)
    if model_class.info.supports_new_data
]


def fitted(name: str, task_type: TaskType) -> tuple[LabModel[Any], NDArray[np.float64]]:
    X, y, new = tiny_dataset(task_type)
    return MODELS.get(name)(seed=1).fit(X, y), new


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_the_onnx_model_predicts_what_the_lab_predicts(name: str, task_type: TaskType) -> None:
    model, new = fitted(name, task_type)

    outputs = run_onnx(model_onnx(model, new.shape[1]), {"features": new})

    if task_type is TaskType.DIMENSIONALITY_REDUCTION:
        np.testing.assert_allclose(
            outputs["projection"], model.transform(new), rtol=RELATIVE, atol=1e-6
        )
    elif task_type.is_classification:
        assert [str(label) for label in outputs["label"]] == [str(c) for c in model.predict(new)]
        if model.info.supports_proba:
            np.testing.assert_allclose(
                outputs["probabilities"], model.predict_proba(new), atol=RELATIVE
            )
    elif task_type is TaskType.REGRESSION:
        np.testing.assert_allclose(
            outputs["prediction"], model.predict(new), rtol=RELATIVE, atol=1e-6
        )
    elif task_type is TaskType.CLUSTERING:
        np.testing.assert_array_equal(outputs["cluster"], model.predict(new))
    else:
        np.testing.assert_array_equal(outputs["anomaly"], model.predict(new))


def test_a_model_that_cannot_place_new_points_says_so() -> None:
    model, new = fitted("tsne", TaskType.DIMENSIONALITY_REDUCTION)

    with pytest.raises(CapabilityError, match="não sabe colocar pontos novos"):
        model_onnx(model, new.shape[1])
