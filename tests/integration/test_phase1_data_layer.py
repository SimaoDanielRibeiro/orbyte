"""Critério da fase 1: carregar o Titanic mostra o perfil correto, os avisos e a sugestão
"classificação binária" (o pandas isolado é confirmado em tests/architecture/).

Verifica-se duas vezes: nos serviços, que a interface aprimorada também vai usar, e na
interface simples, de ponta a ponta.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml_lab.core.services import ColumnKind, DatasetService, Lab, PreparationConfig, TaskType

APP = Path(__file__).resolve().parents[2] / "ui" / "simple" / "app.py"


def test_titanic_profile_warnings_and_task_suggestion() -> None:
    service = DatasetService()
    titanic = service.load_sample("titanic")

    profile = service.profile(titanic.id, target="Survived")
    suggestion = service.suggest_task(titanic.id, "Survived")

    assert (profile.n_rows, profile.n_columns) == (891, 12)
    assert profile.column("Age").n_missing == 177
    assert profile.column("Cabin").n_missing == 687
    assert profile.column("Embarked").n_missing == 2
    assert profile.column("PassengerId").kind is ColumnKind.IDENTIFIER
    assert profile.column("Survived").kind is ColumnKind.BOOLEAN
    assert profile.column("Sex").kind is ColumnKind.CATEGORICAL
    assert profile.column("Name").kind is ColumnKind.TEXT
    assert profile.class_balance is not None
    assert [(c.value, c.count) for c in profile.class_balance.classes] == [("0", 549), ("1", 342)]
    warnings = {(n.code, n.subject) for n in profile.all_notices if n.level == "warning"}
    assert {
        ("identifier_column", "PassengerId"),
        ("missing_values", "Age"),
        ("missing_values", "Cabin"),
    } <= warnings
    assert suggestion.task_type is TaskType.BINARY_CLASSIFICATION


@pytest.fixture
def app_with_titanic() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.selectbox(key="dados.sample").select("titanic").run()
    app.button(key="dados.load").click().run()
    return app


def test_the_data_screen_loads_the_titanic(app_with_titanic: AppTest) -> None:
    app = app_with_titanic

    assert not app.exception
    assert app.subheader[0].value == "Titanic"
    assert [metric.value for metric in app.metric][:2] == ["891", "12"]


def test_the_explore_screen_shows_the_warnings(app_with_titanic: AppTest) -> None:
    app = app_with_titanic.switch_page("pages/explorar.py").run()

    assert not app.exception
    messages = [warning.value for warning in app.warning]
    assert any("Cabin tem 77,1% de valores em falta" in message for message in messages)
    assert any("PassengerId parece um identificador" in message for message in messages)


def test_the_prepare_screen_suggests_binary_classification(app_with_titanic: AppTest) -> None:
    app = app_with_titanic.switch_page("pages/preparar.py").run()

    assert not app.exception
    texts = [markdown.value for markdown in app.markdown]
    assert any("classificação binária" in text for text in texts)
    assert app.multiselect(key=f"preparar.{_titanic_id()}.drop.Survived").value == [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
    ]


def test_the_sidebar_shows_the_dataset_and_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    # O AppTest perde a barra lateral quando a app volta a correr (limitação dos testes do
    # Streamlit com várias páginas), por isso o estado da sessão é montado antes da 1.ª corrida.
    monkeypatch.syspath_prepend(str(APP.parent))
    from components.state import Preparation

    lab = Lab.create()
    titanic = lab.datasets.load_sample("titanic")
    config = PreparationConfig(target="Survived")
    summary = lab.datasets.prepare(titanic.id, config, seed=42)
    app = AppTest.from_file(APP, default_timeout=60)
    app.session_state["lab"] = lab
    app.session_state["dataset_id"] = titanic.id
    app.session_state["preparation"] = Preparation(titanic.id, config, 42, summary)

    app.run()

    sidebar = [markdown.value for markdown in app.sidebar.markdown]
    assert "**Dataset:** Titanic (891 × 12)" in sidebar
    assert "**Tarefa:** Classificação binária" in sidebar


def _titanic_id() -> str:
    return DatasetService().load_sample("titanic").id
