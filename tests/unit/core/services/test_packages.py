"""Os pacotes dos modelos: guardar, pré-visualizar, a biblioteca e carregar (ADR 0039)."""

import io
import json
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.serving import FittedPreparation
from ml_lab.core.foundation.errors import PackageError, RunNotInMemoryError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.schemas.packages import PackageRequest
from ml_lab.core.services.experiments import ExperimentResult
from ml_lab.core.services.lab import Lab
from ml_lab.core.services.packages import PackageFileKind, PackageOrigin, PackageService
from ml_lab.core.tracking import packages as files
from ml_lab.core.tracking.store import InMemoryStore

MOMENT = datetime(2026, 10, 29, 14, 32, 5, tzinfo=UTC)
DROP = CleaningStep("drop_columns", {"columns": ["PassengerId", "Name", "Ticket", "Cabin"]})


def settings(folder: Path) -> Settings:
    return replace(
        get_settings(), artifacts_dir=folder / "artifacts", database_path=folder / "registo.db"
    )


def a_lab(folder: Path) -> Lab:
    """Um laboratório com as pastas e o relógio deste teste (uma instalação)."""
    lab = Lab.create(settings(folder), store=InMemoryStore())
    clock_fixed = PackageService(
        lab.experiments, lab.history, settings(folder), clock=lambda: MOMENT
    )
    return replace(lab, packages=clock_fixed)


def titanic(lab: Lab, model: str = "logistic_regression", **params: Any) -> ExperimentResult:
    sample = lab.datasets.load_sample("titanic")
    config = ExperimentConfig(
        dataset=sample.source,
        preparation=PreparationConfig(target="Survived", cleaning=(DROP,)),
        model=ModelSpec(name=model, hyperparameters=params),
        seed=42,
    )
    return lab.experiments.run(config)


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    return a_lab(tmp_path / "aqui")


def test_the_options_suggest_a_name_and_the_parts(lab: Lab) -> None:
    result = titanic(lab)

    options = lab.packages.options(result.run_id)

    assert options.name == "Titanic — Regressão logística"
    assert options.file_name == "titanic-regressao-logistica.mllab"
    assert options.include_model
    assert options.include_examples  # um dataset de exemplo: as linhas são públicas
    assert options.examples_available
    assert not options.include_training


def test_the_preview_is_the_file_byte_for_byte(lab: Lab) -> None:
    result = titanic(lab)
    request = PackageRequest(include_examples=True, examples=3)

    preview = lab.packages.preview(result.run_id, request)
    saved = lab.packages.save(result.run_id, request)

    _, data = lab.packages.file(saved.package_id)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        inside = {name: archive.read(name) for name in archive.namelist()}
    assert [part.path for part in preview.parts] == list(inside)
    for part in preview.parts:
        assert part.size == len(inside[part.path])
        if part.content is not None:
            assert part.content.encode("utf-8") == inside[part.path]
    assert preview.size == sum(len(content) for content in inside.values())


def test_the_preview_shows_what_each_part_is(lab: Lab) -> None:
    result = titanic(lab)

    preview = lab.packages.preview(result.run_id, PackageRequest())

    parts = {part.path: part for part in preview.parts}
    assert list(parts) == [
        files.README,
        files.MANIFEST,
        files.SIGNATURE,
        files.INPUT,
        files.PREPARATION,
        files.RUN,
        files.MODEL,
        files.EXAMPLES,
    ]
    assert parts[files.README].kind is PackageFileKind.MARKDOWN
    assert parts[files.MODEL].kind is PackageFileKind.BINARY
    assert parts[files.MODEL].content is None
    assert parts[files.README].title == "A ficha do modelo"
    readme = parts[files.README].content or ""
    assert readme.startswith("# Titanic — Regressão logística\n")
    assert "prevê **Survived**" in readme
    assert "## Os limites" in readme
    assert "| Exatidão |" in readme
    codes = {notice.code for notice in preview.notices}
    assert codes == {"package_real_rows", "package_without_onnx"}


def test_optional_parts_come_and_go(lab: Lab) -> None:
    result = titanic(lab, "logistic_regression_gd")

    lean = lab.packages.preview(
        result.run_id, PackageRequest(include_model=False, include_examples=False)
    )
    full = lab.packages.preview(result.run_id, PackageRequest(include_training=True))

    lean_paths = {part.path for part in lean.parts}
    assert files.MODEL not in lean_paths
    assert files.EXAMPLES not in lean_paths
    assert not lean.has_model
    assert "package_without_model" in {notice.code for notice in lean.notices}
    assert {files.ANIMATION, files.MODEL, files.EXAMPLES} <= {part.path for part in full.parts}


def test_the_package_brings_the_preparation_and_the_model_back(lab: Lab) -> None:
    result = titanic(lab, "random_forest", n_estimators=20)
    saved = lab.packages.save(result.run_id, PackageRequest())

    _, data = lab.packages.file(saved.package_id)
    opened = files.read_package(data, lab.packages._key)
    preparation = FittedPreparation.from_documents(
        opened.contents.document(files.INPUT), opened.contents.document(files.PREPARATION)
    )
    model = files.load_model(opened)

    artifacts = lab.experiments.artifacts(result.run_id)
    assert artifacts.dataset.preparation is not None
    raw = artifacts.dataset.preparation.examples.drop(columns=["Survived"])
    features = preparation.transform(raw)
    np.testing.assert_array_equal(features, artifacts.dataset.test.X[: len(raw)])
    np.testing.assert_array_equal(
        model.predict(features), artifacts.model.predict(artifacts.dataset.test.X[: len(raw)])
    )


def test_saving_also_saves_the_run_and_fills_the_library(lab: Lab) -> None:
    result = titanic(lab)

    saved = lab.packages.save(result.run_id, PackageRequest(name="O meu modelo"))

    assert lab.history.entry(result.run_id).saved
    assert (saved.name, saved.file_name) == ("O meu modelo", "o-meu-modelo.mllab")
    assert saved.origin is PackageOrigin.MADE_HERE
    assert (saved.made_here, saved.intact, saved.trusted) == (True, True, True)
    assert saved.run_id == result.run_id
    assert saved.primary_label == "Exatidão"
    assert lab.packages.saved() == (saved,)
    view = lab.packages.contents(saved.package_id)
    assert view.package == saved
    assert view.preview.parts[0].path == files.README
    lab.packages.delete(saved.package_id)
    assert lab.packages.saved() == ()


def test_a_package_made_here_loads_back_into_the_history(lab: Lab, tmp_path: Path) -> None:
    result = titanic(lab)
    saved = lab.packages.save(result.run_id, PackageRequest())
    _, data = lab.packages.file(saved.package_id)
    lab.packages.delete(saved.package_id)

    loaded = lab.packages.load("titanic.mllab", data)

    assert (loaded.origin, loaded.trusted, loaded.made_here) == (PackageOrigin.LOADED, True, True)
    assert loaded.run_id == result.run_id  # é o mesmo run, que já estava no histórico
    assert loaded.notices == ()


def test_a_package_from_elsewhere_waits_for_trust(lab: Lab, tmp_path: Path) -> None:
    result = titanic(lab, "decision_tree")
    _, data = lab.packages.file(lab.packages.save(result.run_id, PackageRequest()).package_id)
    other = a_lab(tmp_path / "noutro-computador")

    loaded = other.packages.load("arvore.mllab", data)

    assert (loaded.made_here, loaded.intact, loaded.trusted) == (False, True, False)
    assert loaded.run_id is None
    assert [notice.code for notice in loaded.notices] == ["package_from_elsewhere"]
    assert other.history.runs() == ()

    trusted = other.packages.trust(loaded.package_id)

    assert trusted.trusted
    assert trusted.run_id is not None
    entry = other.history.entry(trusted.run_id)
    assert entry.saved
    assert entry.name == "Titanic — Árvore de decisão (carregado)"
    opened = other.history.open(trusted.run_id)
    assert opened.notices[0].code == "loaded_from_package"
    # O run volta com o modelo: o Titanic é um exemplo, os dados preparam-se outra vez.
    assert other.experiments.artifacts(trusted.run_id).model is not None


def test_loading_the_same_file_twice_keeps_one_copy(lab: Lab, tmp_path: Path) -> None:
    result = titanic(lab)
    _, data = lab.packages.file(lab.packages.save(result.run_id, PackageRequest()).package_id)
    other = a_lab(tmp_path / "outro")

    first = other.packages.load("a.mllab", data)
    again = other.packages.load("b.mllab", data)

    assert again.package_id == first.package_id
    assert len(other.packages.saved()) == 1


def test_a_package_without_the_model_needs_no_trust(lab: Lab, tmp_path: Path) -> None:
    result = titanic(lab)
    request = PackageRequest(include_model=False)
    _, data = lab.packages.file(lab.packages.save(result.run_id, request).package_id)
    other = a_lab(tmp_path / "outro")

    loaded = other.packages.load("so-a-ficha.mllab", data)

    assert loaded.trusted
    assert not loaded.has_model
    assert loaded.run_id is not None
    assert not other.history.entry(loaded.run_id).saved  # só registado: não há modelo


def test_a_changed_package_never_opens_its_model(lab: Lab, tmp_path: Path) -> None:
    result = titanic(lab)
    _, data = lab.packages.file(lab.packages.save(result.run_id, PackageRequest()).package_id)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        inside = {name: archive.read(name) for name in archive.namelist()}
    run = json.loads(inside[files.RUN])
    run["dataset_name"] = "Outro"
    inside[files.RUN] = json.dumps(run).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in inside.items():
            archive.writestr(name, content)
    other = a_lab(tmp_path / "outro")

    loaded = other.packages.load("alterado.mllab", buffer.getvalue())

    assert not loaded.intact
    assert [notice.code for notice in loaded.notices] == ["package_damaged"]
    with pytest.raises(PackageError, match="estragado ou foi alterado"):
        other.packages.trust(loaded.package_id)


def test_a_run_only_recorded_is_not_a_candidate(lab: Lab, tmp_path: Path) -> None:
    result = titanic(lab)
    later = Lab.create(settings(tmp_path / "aqui"), store=lab.history._store)

    candidates = later.packages.candidates()

    assert [candidate.run_id for candidate in candidates] == [result.run_id]
    assert not candidates[0].available
    assert candidates[0].reason is not None
    assert "reproduz" in candidates[0].reason
    with pytest.raises(RunNotInMemoryError):
        later.packages.options(result.run_id)


def test_memorizing_models_and_projections_are_flagged(lab: Lab) -> None:
    knn = titanic(lab, "knn")
    iris = lab.datasets.load_sample("iris")
    tsne = lab.experiments.run(
        ExperimentConfig(
            dataset=iris.source,
            preparation=PreparationConfig(
                target="species", task_type=TaskType.DIMENSIONALITY_REDUCTION
            ),
            model=ModelSpec(name="tsne"),
            seed=1,
        )
    )

    knn_codes = {n.code for n in lab.packages.preview(knn.run_id, PackageRequest()).notices}
    tsne_preview = lab.packages.preview(tsne.run_id, PackageRequest())

    assert "package_memorized_rows" in knn_codes
    assert "package_cannot_predict" in {notice.code for notice in tsne_preview.notices}
    assert files.EXAMPLES not in {part.path for part in tsne_preview.parts}
