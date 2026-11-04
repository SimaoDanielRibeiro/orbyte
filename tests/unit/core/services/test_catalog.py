from pathlib import Path

import pytest

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    HyperparameterError,
    InvalidParameterError,
)
from ml_lab.core.foundation.types import Partition
from ml_lab.core.services import (
    CatalogService,
    Implementation,
    Lab,
    LaboratoryDemo,
    ModelFamily,
    TaskType,
)
from ml_lab.core.services.catalog import INTRO, doc_sections
from ml_lab.core.tracking.store import InMemoryStore


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
    # O controlo (reinforcement learning) não tem modelos: tem agentes, noutro catálogo.
    for task_type in (task for task in TaskType if task.learns_from_data):
        suggested = catalog.suggested_model(task_type)

        assert suggested is not None
        assert suggested in [card.name for card in catalog.models(task_type)]


def test_a_card_has_everything_the_model_screen_shows(catalog: CatalogService) -> None:
    card = catalog.model("logistic_regression")

    assert card.display_name == "Logistic regression"
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
    assert card.doc.startswith("# Logistic regression")


def test_hyperparameters_are_completed_and_validated(catalog: CatalogService) -> None:
    assert catalog.validate_hyperparameters("logistic_regression", {"C": 0.5}) == {
        "C": 0.5,
        "max_iter": 100,
        "class_weight": "none",
        "fit_intercept": True,
    }
    with pytest.raises(HyperparameterError, match="Maximum iterations"):
        catalog.validate_hyperparameters("logistic_regression", {"max_iter": 1})


def test_unknown_models_get_a_suggestion(catalog: CatalogService) -> None:
    with pytest.raises(ComponentNotFoundError, match="Did you mean 'logistic_regression'"):
        catalog.model("logistic_regresion")


def test_a_missing_didactic_card_gives_an_empty_text(tmp_path: Path) -> None:
    catalog = CatalogService(Settings.from_env({}, home=tmp_path))

    card = catalog.model("logistic_regression")
    assert card.doc == ""
    assert card.sections == ()


# --- a ficha por secções -----------------------------------------------------------------


def test_the_didactic_card_also_comes_split_in_its_sections(catalog: CatalogService) -> None:
    card = catalog.model("knn")

    first = card.sections[0]
    assert (first.key, first.title) == ("intuition", "Intuition")
    assert first.text.startswith('"Tell me who your friends are')
    assert "## " not in "".join(section.text for section in card.sections)
    assert card.sections[1].key == "essential-maths"


def test_text_before_the_first_section_is_the_intro_and_the_title_is_left_out() -> None:
    sections = doc_sections("# KNN\n\nUma frase de entrada.\n\n## Intuição\n\nVizinhos.\n")

    assert [(s.key, s.title, s.text) for s in sections] == [
        (INTRO, "", "Uma frase de entrada."),
        ("intuicao", "Intuição", "Vizinhos."),
    ]


# --- as ligações da página do modelo -----------------------------------------------------


def test_a_card_points_to_its_laboratory_demos_and_its_examples(catalog: CatalogService) -> None:
    knn, ridge, kmeans = (catalog.model(name) for name in ("knn", "ridge", "kmeans"))

    assert knn.laboratory == (LaboratoryDemo.KNN,)
    assert [example.name for example in knn.examples] == ["moons", "circles", "blobs"]
    assert ridge.laboratory == (LaboratoryDemo.REGULARIZATION,)
    assert [example.name for example in ridge.examples] == ["wave", "line"]
    assert kmeans.laboratory == ()
    assert {example.task_type for example in kmeans.examples} == {TaskType.CLUSTERING}


# --- a demonstração -----------------------------------------------------------------------


def test_a_demo_shows_the_model_learning_on_its_first_example(catalog: CatalogService) -> None:
    demo = catalog.demo("logistic_regression_gd")

    assert demo.example.name == "moons"
    assert demo.task_type is TaskType.BINARY_CLASSIFICATION
    assert demo.animation is not None
    assert len(demo.animation.frames) > 1  # treina por passos: vê-se a fronteira a mexer
    assert demo.history
    assert demo.learned.weights is not None
    assert demo.reports[Partition.TEST].metrics
    assert demo.charts.roc


def test_the_example_decides_the_task(catalog: CatalogService) -> None:
    assert catalog.demo("knn", example="blobs").task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert catalog.demo("kmeans").task_type is TaskType.CLUSTERING
    assert catalog.demo("pca").embedding is not None


def test_a_hyperparameter_changes_what_the_demo_shows(catalog: CatalogService) -> None:
    one = catalog.demo("knn", {"n_neighbors": 1})
    many = catalog.demo("knn", {"n_neighbors": 50})

    # Com um só vizinho, cada ponto de treino é o seu próprio vizinho: acerta-os todos.
    assert one.reports[Partition.TRAIN].metric("accuracy").value == 1.0
    smoothed = many.reports[Partition.TRAIN].metric("accuracy").value
    assert smoothed is not None
    assert smoothed < 1.0
    assert one.hyperparameters["n_neighbors"] == 1


def test_the_same_demo_is_trained_only_once(catalog: CatalogService) -> None:
    first = catalog.demo("decision_tree")

    # Os valores por omissão escritos à mão são a mesma demonstração.
    assert catalog.demo("decision_tree", catalog.model("decision_tree").defaults) is first


def test_an_example_of_another_task_is_refused_with_the_choices(catalog: CatalogService) -> None:
    with pytest.raises(InvalidParameterError, match="moons, circles, blobs"):
        catalog.demo("knn", example="wave")


def test_invalid_hyperparameters_are_refused_before_training(catalog: CatalogService) -> None:
    with pytest.raises(HyperparameterError):
        catalog.demo("knn", {"n_neighbors": 0})


def test_the_metrics_catalog_explains_each_metric_with_its_formula(
    catalog: CatalogService,
) -> None:
    everything = catalog.metrics()
    regression = {card.name for card in catalog.metrics(TaskType.REGRESSION)}

    assert len(everything) == len({card.name for card in everything}) >= 20
    assert regression == {"r2", "rmse", "mae", "mse"}
    precision = next(card for card in everything if card.name == "precision")
    assert precision.as_percent
    assert precision.higher_is_better
    assert r"\mathrm{TP}" in precision.formula
    assert [symbol.symbol for symbol in precision.symbols] == [r"\mathrm{TP}", r"\mathrm{FP}"]
    assert next(card for card in everything if card.name == "roc_auc").needs_proba


def test_a_demo_never_reaches_the_history() -> None:
    lab = Lab.create(store=InMemoryStore())

    lab.catalog.demo("logistic_regression")

    assert not lab.history.runs()
    assert not lab.datasets.loaded()
