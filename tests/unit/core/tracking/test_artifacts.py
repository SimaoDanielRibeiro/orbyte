"""Os ficheiros dos runs guardados: o modelo (joblib) e os documentos do treino (JSON)."""

from pathlib import Path

import numpy as np
import pytest

from ml_lab.core.foundation.errors import TrackingError
from ml_lab.core.models.base import MODELS
from ml_lab.core.tracking.artifacts import MODEL_FILE, ArtifactStore


def test_a_saved_model_predicts_the_same_after_loading(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 2)), np.repeat([0, 1], 20)
    model = MODELS.get("logistic_regression")().fit(X, y)
    files = ArtifactStore(tmp_path)

    files.save("run-a", model, {"animation": {"frames": [1, 2]}, "network": [0.5]})
    loaded = files.model("run-a")

    assert (tmp_path / "run-a" / MODEL_FILE).is_file()
    assert (tmp_path / "run-a" / "animation.json").is_file()
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert files.document("run-a", "animation") == {"frames": [1, 2]}
    assert files.document("run-a", "network") == [0.5]
    assert files.has_model("run-a")


def test_a_run_without_documents_or_files(tmp_path: Path) -> None:
    files = ArtifactStore(tmp_path)
    files.save("run-a", {"modelo": "qualquer objeto"})

    assert files.document("run-a", "animation") is None
    assert not files.has_model("run-b")
    with pytest.raises(TrackingError, match="run-b"):
        files.model("run-b")


def test_deleting_removes_the_folder(tmp_path: Path) -> None:
    files = ArtifactStore(tmp_path)
    files.save("run-a", [1, 2, 3])

    files.delete("run-a")
    files.delete("run-a")  # apagar outra vez não é um erro

    assert not (tmp_path / "run-a").exists()


def test_names_cannot_reach_outside_the_folder(tmp_path: Path) -> None:
    files = ArtifactStore(tmp_path / "runs")

    with pytest.raises(TrackingError, match="Invalid"):
        files.save("../fora", [1])
    with pytest.raises(TrackingError, match="Invalid"):
        files.delete("..")
    with pytest.raises(TrackingError, match="Invalid"):
        files.save("run-a", [1], {"../fora": {}})
