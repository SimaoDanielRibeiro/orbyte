import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.io.base import LoadedTable, SourceSpec
from ml_lab.core.data.lineage import Stage
from ml_lab.core.data.preparation import PreparationConfig, prepare, task_options
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import Partition, TaskType

DROP_TEXT = CleaningStep("drop_columns", {"columns": ["PassengerId", "Name"]})


def table(frame: pd.DataFrame) -> LoadedTable:
    return LoadedTable("Passageiros", frame, SourceSpec("test", {}))


def classification(**overrides: object) -> PreparationConfig:
    settings: dict[str, object] = {
        "target": "Survived",
        "cleaning": (DROP_TEXT,),
        "split": SplitConfig(SplitStrategy.STRATIFIED, test_size=0.25),
    }
    settings.update(overrides)
    return PreparationConfig(**settings)  # type: ignore[arg-type]


def test_preparation_delivers_numpy_arrays_with_metadata(passengers: pd.DataFrame) -> None:
    prepared = prepare(table(passengers), classification(), seeds=SeedManager(42))
    dataset = prepared.dataset

    assert dataset.task_type is TaskType.BINARY_CLASSIFICATION
    assert dataset.class_labels == ("0", "1")
    assert dataset.train.X.dtype == np.float64
    assert dataset.train.y is not None
    assert dataset.train.y.dtype == np.int64
    assert (len(dataset.train), len(dataset.test)) == (15, 5)
    assert dataset.validation is None
    assert dataset.feature_names == (
        "Pclass",
        "Sex_female",
        "Sex_male",
        "Age",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
        "Fare",
    )
    assert dataset.n_features == dataset.train.X.shape[1] == 8
    assert not np.isnan(dataset.train.X).any()


def test_the_lineage_goes_from_input_to_output_in_order(passengers: pd.DataFrame) -> None:
    lineage = prepare(table(passengers), classification(), seeds=SeedManager(1)).dataset.lineage

    stages = [step.stage for step in lineage]

    assert stages[0] is Stage.INPUT
    assert stages[-1] is Stage.OUTPUT
    assert stages == sorted(stages, key=list(Stage).index)
    assert lineage.for_stage(Stage.CLEANING)[0].columns_removed == ("PassengerId", "Name")


def test_nothing_is_learned_from_the_test_rows(passengers: pd.DataFrame) -> None:
    prepared = prepare(table(passengers), classification(), seeds=SeedManager(3))
    dataset = prepared.dataset
    train_rows = passengers.loc[dataset.train.rows]
    age = dataset.feature_names.index("Age")

    # Depois da padronização, a média da idade no treino é 0; no teste, normalmente não.
    assert dataset.train.X[:, age].mean() == pytest.approx(0.0)
    step = next(s for s in dataset.lineage if s.operation == "impute")
    assert f"{train_rows['Age'].median():.2f}" in step.description


def test_the_same_seed_gives_the_same_split(passengers: pd.DataFrame) -> None:
    first = prepare(table(passengers), classification(), seeds=SeedManager(7)).dataset
    again = prepare(table(passengers), classification(), seeds=SeedManager(7)).dataset
    other = prepare(table(passengers), classification(), seeds=SeedManager(8)).dataset

    assert np.array_equal(first.test.rows, again.test.rows)
    assert np.array_equal(first.test.X, again.test.X)
    assert not np.array_equal(first.test.rows, other.test.rows)


def test_rows_point_back_to_the_original_table(passengers: pd.DataFrame) -> None:
    dataset = prepare(table(passengers), classification(), seeds=SeedManager(0)).dataset

    for split in dataset.splits.values():
        assert split.y is not None
        expected = passengers.loc[split.rows, "Survived"].to_numpy()
        assert np.array_equal(split.y, expected)


def test_rows_without_target_are_removed_and_recorded(passengers: pd.DataFrame) -> None:
    frame = passengers.assign(Survived=passengers["Survived"].astype("float64"))
    frame.loc[[0, 1], "Survived"] = np.nan

    prepared = prepare(table(frame), classification(split=SplitConfig()), seeds=SeedManager(0))

    step = next(s for s in prepared.dataset.lineage if s.operation == "drop_missing_target")
    assert step.rows_removed == 2
    assert len(prepared.cleaned) == 18
    assert "missing_target" in [notice.code for notice in prepared.notices]


def test_unsupervised_preparation_has_no_target(passengers: pd.DataFrame) -> None:
    config = PreparationConfig(cleaning=(DROP_TEXT,))

    dataset = prepare(table(passengers), config, seeds=SeedManager(0)).dataset

    assert dataset.task_type is TaskType.CLUSTERING
    assert dataset.train.y is None
    assert "Survived" in dataset.feature_names


def test_a_validation_split_is_transformed_like_the_test(passengers: pd.DataFrame) -> None:
    config = classification(split=SplitConfig(test_size=0.2, validation_size=0.2))

    dataset = prepare(table(passengers), config, seeds=SeedManager(0)).dataset

    assert dataset.validation is not None
    assert list(dataset.splits) == [Partition.TRAIN, Partition.VALIDATION, Partition.TEST]
    assert dataset.validation.X.shape[1] == dataset.n_features


def test_the_projection_is_part_of_the_dataset(passengers: pd.DataFrame) -> None:
    dataset = prepare(table(passengers), classification(), seeds=SeedManager(0)).dataset

    assert dataset.projection is not None
    assert dataset.projection.coordinates[Partition.TEST].shape == (5, 2)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"task_type": TaskType.REGRESSION}, InvalidParameterError, "doesn't suit the target"),
        (
            {"target": None, "task_type": TaskType.BINARY_CLASSIFICATION},
            InvalidParameterError,
            "needs a target column",
        ),
        (
            {
                "cleaning": (
                    DROP_TEXT,
                    CleaningStep(
                        "drop_columns",
                        {"columns": ["Pclass", "Sex", "Age", "Embarked", "Fare", "Cabin"]},
                    ),
                )
            },
            DataError,
            "No feature the model can use",
        ),
        (
            {"cleaning": (CleaningStep("drop_missing_rows", {"how": "any"}),)},
            DataError,
            "No rows were left",
        ),
    ],
)
def test_impossible_configurations_are_explained(
    passengers: pd.DataFrame, overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    frame = passengers.assign(Cabin=np.nan) if "cleaning" in overrides else passengers

    with pytest.raises(error, match=message):
        prepare(table(frame), classification(**overrides), seeds=SeedManager(0))


def test_stratification_needs_classes(numbers: pd.DataFrame) -> None:
    config = PreparationConfig(target="y", split=SplitConfig(SplitStrategy.STRATIFIED))

    with pytest.raises(InvalidParameterError, match="needs classes"):
        prepare(table(numbers), config, seeds=SeedManager(0))


def test_a_reference_column_stratifies_like_a_target(passengers: pd.DataFrame) -> None:
    # A referência não entra no treino, mas reparte as linhas como o alvo repartiria.
    supervised = prepare(table(passengers), classification(), seeds=SeedManager(0)).dataset
    clustering = prepare(
        table(passengers), classification(task_type=TaskType.CLUSTERING), seeds=SeedManager(0)
    ).dataset

    assert clustering.task_type is TaskType.CLUSTERING
    assert np.array_equal(clustering.test.rows, supervised.test.rows)
    assert "Survived" not in clustering.feature_names


def test_task_options_say_what_the_chosen_task_changes() -> None:
    clustering = task_options(TaskType.CLUSTERING, "species")
    regression = task_options(TaskType.REGRESSION, "price")
    without_target = task_options(TaskType.CLUSTERING, None)

    assert clustering.reference == "species"
    assert clustering.reference_note is not None
    assert "doesn't go into the training" in clustering.reference_note
    assert clustering.split_strategies[0] is SplitStrategy.STRATIFIED  # há classes
    assert regression.reference is None
    assert SplitStrategy.STRATIFIED not in regression.split_strategies
    assert without_target.reference_note is None
    assert SplitStrategy.STRATIFIED not in without_target.split_strategies


def test_in_anomaly_detection_the_rarest_reference_class_is_number_one() -> None:
    # Por ordem alfabética, «anomalia» seria a classe 0; por ser a mais rara, fica com o 1
    # em todas as partes, que é o que "1 = anomalia" quer dizer nos avaliadores.
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=40), "kind": ["anomalia"] * 6 + ["normal"] * 34})
    config = PreparationConfig(target="kind", task_type=TaskType.ANOMALY_DETECTION)

    dataset = prepare(table(frame), config, seeds=SeedManager(0)).dataset

    assert dataset.class_labels == ("normal", "anomalia")
    assert sum(int(split.y.sum()) for split in dataset.splits.values() if split.y is not None) == 6


def test_regression_targets_stay_as_numbers(numbers: pd.DataFrame) -> None:
    config = PreparationConfig(target="y", transforms=TransformConfig(scaling=None))

    dataset = prepare(table(numbers), config, seeds=SeedManager(0)).dataset

    assert dataset.task_type is TaskType.REGRESSION
    assert dataset.class_labels == ()
    assert dataset.train.y is not None
    assert dataset.train.y.dtype == np.float64
    assert np.allclose(dataset.train.y, 3 * dataset.train.X[:, 0] - 2 * dataset.train.X[:, 1] + 1)
