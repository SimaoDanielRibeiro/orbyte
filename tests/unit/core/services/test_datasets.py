from pathlib import Path

import pytest

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import (
    DataSourceError,
    DataValidationError,
    InvalidParameterError,
)
from ml_lab.core.services import (
    DatasetService,
    Partition,
    PreparationConfig,
    SourceSpec,
    SplitConfig,
    SplitStrategy,
    TaskType,
)
from ml_lab.core.services.datasets import MAX_PROJECTION_POINTS


@pytest.fixture
def service(tmp_path: Path) -> DatasetService:
    return DatasetService(Settings.from_env({}, home=tmp_path))


def test_catalogs_list_samples_generators_and_operations(service: DatasetService) -> None:
    assert [sample.name for sample in service.samples()] == [
        "california_housing",
        "iris",
        "titanic",
        "wine",
    ]
    assert {g.name for g in service.generators()} == {
        "anomalies",
        "blobs",
        "circles",
        "moons",
        "regression",
    }
    assert "drop_duplicates" in {operation.name for operation in service.cleaning_operations()}


def test_loading_the_same_source_twice_reuses_it(service: DatasetService) -> None:
    first = service.load_sample("iris")
    again = service.load(SourceSpec("sample", {"name": "iris"}))

    assert first is again
    assert first.n_rows == 150
    assert first.suggested_target == "species"
    assert [summary.id for summary in service.loaded()] == [first.id]


def test_generated_datasets_get_the_same_id_with_or_without_defaults(
    service: DatasetService,
) -> None:
    implicit = service.generate("moons", {}, seed=3)
    explicit = service.generate("moons", {"n_samples": 200, "noise": 0.2}, seed=3)

    assert implicit.id == explicit.id
    assert implicit.source.params["parameters"] == {"n_samples": 200, "noise": 0.2}


def test_uploads_are_saved_with_a_content_hash(service: DatasetService, tmp_path: Path) -> None:
    content = b"idade,nome\n30,Ana\n41,Rui\n"

    summary = service.upload("../os meus dados.csv", content)
    again = service.upload("os meus dados.csv", content)

    saved = list((tmp_path / "data" / "uploads").iterdir())
    assert [path.name for path in saved] == ["os_meus_dados-" + saved[0].name[-12:]]
    assert saved[0].read_bytes() == content
    assert summary.id == again.id
    assert summary.name == "os meus dados.csv"


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [("dados.json", b"{}", DataSourceError), ("vazio.csv", b"", DataValidationError)],
)
def test_bad_uploads_are_refused_before_being_saved(
    service: DatasetService, tmp_path: Path, filename: str, content: bytes, error: type[Exception]
) -> None:
    with pytest.raises(error):
        service.upload(filename, content)

    assert not (tmp_path / "data" / "uploads").exists()


def test_profile_task_and_cleaning_suggestions(service: DatasetService) -> None:
    titanic = service.load_sample("titanic")

    profile = service.profile(titanic.id, target="Survived")
    suggestion = service.suggest_task(titanic.id, "Survived")
    suggestions = service.suggest_cleaning(titanic.id, "Survived")

    assert profile is service.profile(titanic.id, target="Survived")  # fica em cache
    assert suggestion.task_type is TaskType.BINARY_CLASSIFICATION
    assert [s.column for s in suggestions] == ["PassengerId", "Name", "Ticket", "Cabin"]
    assert service.preview(titanic.id, n_rows=3).n_rows == 891


def test_prepare_summarizes_the_result_for_the_screen(service: DatasetService) -> None:
    iris = service.load_sample("iris")
    config = PreparationConfig(target="species", split=SplitConfig(SplitStrategy.STRATIFIED))

    summary = service.prepare(iris.id, config, seed=42)

    assert summary.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert [part.partition for part in summary.partitions] == [Partition.TRAIN, Partition.TEST]
    assert sum(part.n_rows for part in summary.partitions) == 150
    train = summary.partitions[0]
    assert [value.count for value in train.classes] == [40, 40, 40]
    assert summary.projection is not None
    assert summary.projection.method == "pca"
    assert len(summary.projection.x) == len(summary.projection.labels) == 150
    assert summary.transformed_preview.columns == summary.feature_names


def test_big_projections_are_thinned_out(service: DatasetService) -> None:
    housing = service.load_sample("california_housing")
    config = PreparationConfig(target="median_house_value")

    projection = service.prepare(housing.id, config, seed=0).projection

    assert projection is not None
    assert len(projection.x) <= MAX_PROJECTION_POINTS
    assert len(projection.values) == len(projection.x)


def test_prepared_data_is_reused_for_the_same_configuration(service: DatasetService) -> None:
    iris = service.load_sample("iris")
    config = PreparationConfig(target="species")

    first = service.prepared(iris.source, config, seed=1)

    assert service.prepared(iris.source, config, seed=1) is first
    assert service.prepared(iris.source, config, seed=2) is not first


def test_unknown_datasets_are_explained(service: DatasetService) -> None:
    with pytest.raises(InvalidParameterError, match="carrega-o primeiro"):
        service.summary("nada")


def test_old_datasets_leave_memory_first(tmp_path: Path) -> None:
    service = DatasetService(Settings.from_env({}, home=tmp_path), max_datasets=1)
    iris = service.load_sample("iris")

    service.load_sample("wine")

    with pytest.raises(InvalidParameterError):
        service.summary(iris.id)
