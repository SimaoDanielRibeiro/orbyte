"""Contrato da página de cada modelo: a demonstração, a ficha por secções e as ligações.

Cada modelo do registo mostra-se a trabalhar num conjunto de exemplo (ADR 0028), e a sua ficha
tem as secções do modelo das fichas: um modelo novo que falhe aqui não tem página.
"""

import pytest

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS
from ml_lab.core.services.catalog import CatalogService
from ml_lab.core.services.demos import EXAMPLES
from ml_lab.core.services.laboratory import LaboratoryDemo
from ml_lab.core.training.step_metrics import STEP_METRICS

SECTIONS = (
    "intuition",
    "essential-maths",
    "hyperparameters",
    "when-to-use",
    "pitfalls",
    "what-to-watch-in-training",
)
"""As secções do modelo das fichas (``docs/algorithms/README.md``), por esta ordem."""


@pytest.fixture(scope="module")
def catalog() -> CatalogService:
    return CatalogService()


@pytest.mark.parametrize("name", MODELS.names())
def test_every_model_shows_itself_working_on_an_example(catalog: CatalogService, name: str) -> None:
    card = catalog.model(name)

    demo = catalog.demo(name)

    assert demo.example == card.examples[0]
    assert demo.task_type is demo.example.task_type
    assert demo.task_type in card.task_types
    assert demo.charts.task_type is demo.task_type
    assert demo.reports[demo.main_partition].metrics
    if demo.task_type is TaskType.DIMENSIONALITY_REDUCTION:
        assert demo.embedding is not None  # o resultado é o mapa dos pontos
    else:
        assert demo.animation is not None
        assert demo.animation.how_to_read, "o campo diz como se lê"
        if card.iterative:
            assert len(demo.animation.frames) > 1, "um modelo que treina por passos anima-se"
    # As métricas de cada passo têm nome a mostrar no vocabulário (os gráficos do treino).
    reported = {name for record in demo.history for name in record.metrics}
    assert reported <= STEP_METRICS.keys(), f"sem nome: {reported - STEP_METRICS.keys()}"


@pytest.mark.parametrize("name", MODELS.names())
def test_every_didactic_card_has_the_sections_of_the_template(
    catalog: CatalogService, name: str
) -> None:
    sections = catalog.model(name).sections

    assert tuple(section.key for section in sections) == SECTIONS
    assert all(section.text for section in sections)


def test_every_example_has_models_that_use_it() -> None:
    tasks = {task for _, model in MODELS.items() for task in model.info.task_types}

    assert {example.info.task_type for example in EXAMPLES} <= tasks
    names = [example.info.name for example in EXAMPLES]
    assert len(set(names)) == len(names)


def test_every_laboratory_demo_points_to_models_that_exist() -> None:
    for demo in LaboratoryDemo:
        assert demo.models
        assert demo.models <= set(MODELS.names())
