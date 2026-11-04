"""Esquemas Pydantic: hiperparâmetros, configuração da experiência e mensagens que explicam."""

import doctest
import json
from typing import Literal

import pytest
from pydantic import Field, ValidationError

import ml_lab.core.schemas.experiment as experiment_module
from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.transforms.scaling import ScalingMethod
from ml_lab.core.foundation.errors import HyperparameterError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.supervised.classification.logistic_regression import (
    LogisticRegressionParams,
)
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.schemas.validation import describe_errors


class Example(Hyperparameters):
    """Um hiperparâmetro de cada tipo, com limites."""

    rate: float = Field(default=0.5, gt=0, le=1, title="Taxa")
    count: int = Field(default=3, ge=1, lt=10, title="Contagem")
    mode: Literal["a", "b"] = Field(default="a", title="Modo")
    active: bool = Field(default=True, title="Ativo")


def iris_config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "dataset": SourceSpec("sample", {"name": "iris"}),
        "preparation": PreparationConfig(
            target="species",
            cleaning=(CleaningStep("drop_duplicates"),),
            split=SplitConfig(SplitStrategy.STRATIFIED),
        ),
        "model": ModelSpec(name="logistic_regression", hyperparameters={"C": 0.5}),
        "seed": 42,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


# --- hiperparâmetros -------------------------------------------------------------------


def test_missing_values_take_their_defaults() -> None:
    assert Example.parse({}) == Example()
    assert Example.parse({"rate": 0.25}).rate == 0.25


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"rate": 0}, "Taxa must be greater than 0."),
        ({"rate": 2}, "Taxa can't be more than 1."),
        ({"count": 0}, "Contagem must be at least 1."),
        ({"count": 10}, "Contagem must be less than 10."),
        ({"count": "x"}, "Contagem must be a whole number; got 'x'."),
        ({"mode": "c"}, "Modo must be one of: 'a' or 'b'."),
        ({"active": "talvez"}, "Ativo must be true or false; got 'talvez'."),
        ({"speed": 1}, "speed isn't a parameter of this model."),
    ],
)
def test_each_problem_is_explained_in_plain_words(values: dict[str, object], message: str) -> None:
    with pytest.raises(HyperparameterError) as error:
        Example.parse(values, model="exemplo")

    assert error.value.problems == (message,)
    assert str(error.value).startswith("Invalid hyperparameters for 'exemplo':")


def test_hyperparameters_are_immutable() -> None:
    with pytest.raises(ValidationError):
        Example().rate = 0.1  # type: ignore[misc]


def test_titles_and_form_schema() -> None:
    schema = LogisticRegressionParams.form_schema()

    assert Example.titles() == {
        "rate": "Taxa",
        "count": "Contagem",
        "mode": "Modo",
        "active": "Ativo",
    }
    assert schema["properties"]["C"]["x-scale"] == "log"
    assert schema["properties"]["class_weight"]["x-labels"]["balanced"] == "Balanced"
    assert schema["properties"]["max_iter"]["minimum"] == 10


# --- configuração da experiência ---------------------------------------------------


def test_the_configuration_goes_to_json_and_back_unchanged() -> None:
    config = iris_config()

    restored = ExperimentConfig.model_validate_json(config.model_dump_json())

    assert restored == config
    assert restored.preparation.split.strategy is SplitStrategy.STRATIFIED
    assert restored.preparation.cleaning == (CleaningStep("drop_duplicates"),)
    assert isinstance(restored.dataset, SourceSpec)


def test_json_values_are_turned_into_the_right_types() -> None:
    config = ExperimentConfig.model_validate(
        {
            "dataset": {"kind": "sample", "params": {"name": "titanic"}},
            "preparation": {
                "target": "Survived",
                "task_type": "binary_classification",
                "transforms": {"scaling": "minmax"},
            },
            "model": {"name": "logistic_regression"},
        }
    )

    assert config.preparation.task_type is TaskType.BINARY_CLASSIFICATION
    assert config.preparation.transforms.scaling is ScalingMethod.MINMAX
    assert config.seed is None
    assert config.model.hyperparameters == {}
    json.dumps(config.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seed": -1}, "seed must be at least 0."),
        ({"seed": 2**32}, "seed can't be more than 4294967295."),
        ({"model": {"name": ""}}, "model.name"),
    ],
)
def test_invalid_configurations_are_explained(change: dict[str, object], message: str) -> None:
    values = iris_config().model_dump(mode="json") | change

    with pytest.raises(ValidationError) as error:
        ExperimentConfig.model_validate(values)

    assert message in " ".join(describe_errors(error.value))


def test_errors_from_the_data_layer_keep_their_own_message() -> None:
    values = iris_config().model_dump(mode="json")
    values["preparation"]["split"]["test_size"] = 1.5

    with pytest.raises(ValidationError) as error:
        ExperimentConfig.model_validate(values)

    (message,) = describe_errors(error.value)
    assert message.startswith("The test share must be between 0 and 1")


def test_missing_fields_are_named() -> None:
    with pytest.raises(ValidationError) as error:
        ExperimentConfig.model_validate({"dataset": {"kind": "sample"}})

    assert "model is missing a value." in describe_errors(error.value)


def test_the_configuration_is_immutable() -> None:
    with pytest.raises(ValidationError):
        iris_config().seed = 1  # type: ignore[misc]


def test_docstring_examples_run_as_documented() -> None:
    result = doctest.testmod(experiment_module)

    assert result.attempted > 0
    assert result.failed == 0
