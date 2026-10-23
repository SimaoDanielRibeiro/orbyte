"""Critério da fase 2: no Iris, escolher o modelo, treinar e ver as métricas na interface.

Verifica-se nos serviços (o fluxo que a API também vai usar) e na interface simples,
de ponta a ponta: Dados → Preparar → Modelo → Avaliar.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml_lab.core.services import (
    ExperimentConfig,
    Lab,
    ModelSpec,
    Partition,
    PreparationConfig,
    SplitConfig,
    SplitStrategy,
    TaskType,
)

APP = Path(__file__).resolve().parents[2] / "ui" / "simple" / "app.py"


def test_logistic_regression_learns_the_iris() -> None:
    lab = Lab.create()
    iris = lab.datasets.load_sample("iris")
    preparation = PreparationConfig(target="species", split=SplitConfig(SplitStrategy.STRATIFIED))

    cards = lab.catalog.models(lab.datasets.suggest_task(iris.id, "species").task_type)
    result = lab.experiments.run(
        ExperimentConfig(
            dataset=iris.source,
            preparation=preparation,
            model=ModelSpec(name=cards[0].name),
            seed=42,
        )
    )

    assert [card.name for card in cards] == ["logistic_regression"]
    assert result.task_type is TaskType.MULTICLASS_CLASSIFICATION
    test = result.reports[Partition.TEST]
    accuracy = test.metric("accuracy").value
    assert accuracy is not None
    assert accuracy >= 0.9
    assert test.confusion_matrix is not None
    assert sum(map(sum, test.confusion_matrix.counts)) == 30


@pytest.fixture
def trained_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.selectbox(key="dados.sample").select("iris").run()
    app.button(key="dados.load").click().run()
    app.switch_page("pages/preparar.py").run()
    app.switch_page("pages/modelo.py").run()
    app.button(key="modelo.train").click().run()
    return app


def test_the_interface_trains_and_shows_the_metrics(trained_app: AppTest) -> None:
    app = trained_app

    assert not app.exception
    assert app.title[0].value == "Avaliar"
    cards = {metric.label: metric.value for metric in app.metric}
    assert cards["Exatidão"] == "90,0%"
    assert "Exatidão equilibrada" in cards
    expanders = [expander.label for expander in app.expander]
    assert any(label.startswith("Exatidão: ") for label in expanders)


def test_the_model_screen_builds_the_form_from_the_schema() -> None:
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.selectbox(key="dados.sample").select("iris").run()
    app.button(key="dados.load").click().run()
    app.switch_page("pages/preparar.py").run()

    app.switch_page("pages/modelo.py").run()

    assert not app.exception
    assert app.selectbox(key="modelo.multiclass_classification").value == "logistic_regression"
    assert app.select_slider(key="modelo.logistic_regression.C").value == 1.0
    assert app.slider(key="modelo.logistic_regression.max_iter").value == 100
    assert app.selectbox(key="modelo.logistic_regression.class_weight").value == "none"
    assert app.toggle(key="modelo.logistic_regression.fit_intercept").value is True


def test_without_a_preparation_the_model_screen_explains_what_to_do() -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()

    app.switch_page("pages/modelo.py").run()

    assert not app.exception
    assert "Prepara primeiro os dados" in app.info[0].value


def test_the_sidebar_shows_the_model_and_the_last_result() -> None:
    # O AppTest perde a barra lateral quando a app volta a correr (ver o teste da fase 1),
    # por isso o estado da sessão é montado antes da primeira corrida.
    lab = Lab.create()
    iris = lab.datasets.load_sample("iris")
    result = lab.experiments.run(
        ExperimentConfig(
            dataset=iris.source,
            preparation=PreparationConfig(target="species"),
            model=ModelSpec(name="logistic_regression"),
            seed=42,
        )
    )
    app = AppTest.from_file(APP, default_timeout=60)
    app.session_state["lab"] = lab
    app.session_state["model"] = lab.catalog.model("logistic_regression")
    app.session_state["result"] = result

    app.run()

    sidebar = [markdown.value for markdown in app.sidebar.markdown]
    assert "**Modelo:** Regressão logística" in sidebar
    assert result.primary is not None
    assert f"**Último resultado:** Exatidão no teste: {result.primary.formatted}" in sidebar
