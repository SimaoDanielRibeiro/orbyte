from pathlib import Path

import pytest

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import ComponentNotFoundError, HyperparameterError
from ml_lab.core.services import CatalogService, Implementation, ModelFamily, TaskType


@pytest.fixture
def catalog() -> CatalogService:
    return CatalogService()


def test_the_catalog_is_filtered_by_task(catalog: CatalogService) -> None:
    assert [card.name for card in catalog.models(TaskType.BINARY_CLASSIFICATION)] == [
        "logistic_regression"
    ]
    assert catalog.models(TaskType.REGRESSION) == []
    assert [card.name for card in catalog.models()] == ["logistic_regression"]


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
