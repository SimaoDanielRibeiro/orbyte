from pathlib import Path

import pytest

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import ComponentNotFoundError, HyperparameterError
from ml_lab.core.services import CatalogService, Implementation, ModelFamily, TaskType


@pytest.fixture
def catalog() -> CatalogService:
    return CatalogService()


def test_the_catalog_is_filtered_by_task(catalog: CatalogService) -> None:
    binary = [card.name for card in catalog.models(TaskType.BINARY_CLASSIFICATION)]
    regression = [card.name for card in catalog.models(TaskType.REGRESSION)]
    everything = [card.name for card in catalog.models()]

    assert "logistic_regression" in binary
    assert "logistic_regression" not in regression
    assert "linear_regression_gd" in regression
    assert set(everything) >= set(binary) | set(regression)
    assert all(
        TaskType.REGRESSION in card.task_types for card in catalog.models(TaskType.REGRESSION)
    )


def test_the_catalog_groups_the_models_by_family(catalog: CatalogService) -> None:
    cards = catalog.models()

    keys = [(card.family.order, card.display_name) for card in cards]
    assert keys == sorted(keys)
    assert cards[0].family is ModelFamily.LINEAR  # pela ordem do enum, não do nome


def test_every_task_has_a_suggested_first_model(catalog: CatalogService) -> None:
    for task_type in TaskType:
        suggested = catalog.suggested_model(task_type)

        assert suggested is not None
        assert suggested in [card.name for card in catalog.models(task_type)]


def test_a_card_has_everything_the_model_screen_shows(catalog: CatalogService) -> None:
    card = catalog.model("logistic_regression")

    assert card.display_name == "Regressão logística"
    assert card.family is ModelFamily.LINEAR
    assert card.implementation is Implementation.SKLEARN
    assert card.supports_proba
    assert not card.iterative
    assert card.task_types == (
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    )
    assert card.defaults == {
        "C": 1.0,
        "max_iter": 100,
        "class_weight": "none",
        "fit_intercept": True,
    }
    assert set(card.hyperparameters_schema["properties"]) == set(card.defaults)
    assert card.doc.startswith("# Regressão logística")


def test_hyperparameters_are_completed_and_validated(catalog: CatalogService) -> None:
    assert catalog.validate_hyperparameters("logistic_regression", {"C": 0.5}) == {
        "C": 0.5,
        "max_iter": 100,
        "class_weight": "none",
        "fit_intercept": True,
    }
    with pytest.raises(HyperparameterError, match="Máximo de iterações"):
        catalog.validate_hyperparameters("logistic_regression", {"max_iter": 1})


def test_unknown_models_get_a_suggestion(catalog: CatalogService) -> None:
    with pytest.raises(ComponentNotFoundError, match="Querias dizer 'logistic_regression'"):
        catalog.model("logistic_regresion")


def test_a_missing_didactic_card_gives_an_empty_text(tmp_path: Path) -> None:
    catalog = CatalogService(Settings.from_env({}, home=tmp_path))

    assert catalog.model("logistic_regression").doc == ""
