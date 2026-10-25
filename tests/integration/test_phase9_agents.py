"""Critério da fase 9: os agentes encontram um modelo para o Titanic sem tocar no teste.

Pelos serviços: o agente de dados prepara o Titanic e diz porquê; a triagem passa por uma
família de cada vez, com os valores por omissão; só os finalistas são afinados; tudo se mede
com validação cruzada nos folds do treino. O vencedor é o único run que usa o teste: fica no
registo em SQLite, e outro ``Lab`` (como reabrir a app) abre-o.

Na interface: o ecrã Dados oferece os dois modos, o ecrã Agentes faz o pedido e mostra o
quadro, o vencedor abre-se no Avaliar e um candidato treina-se à mão no Treinar.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml_lab.config import Settings, get_settings
from ml_lab.core.services import (
    AutoMLRequest,
    Lab,
    ModelFamily,
    SearchPhase,
    SearchSettings,
    SourceSpec,
)

APP = Path(__file__).resolve().parents[2] / "ui" / "simple" / "app.py"
TITANIC = SourceSpec("sample", {"name": "titanic"})


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Uma base de dados e uma pasta de artefactos vazias, só deste teste (também para a app)."""
    monkeypatch.setenv("ML_LAB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ML_LAB_DATABASE_PATH", str(tmp_path / "artifacts" / "ml_lab.db"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def test_the_agents_find_a_model_for_the_titanic_without_touching_the_test(
    fresh: Settings,
) -> None:
    lab = Lab.create(fresh)
    request = AutoMLRequest(
        dataset=TITANIC,
        target="Survived",
        settings=SearchSettings(budget=15, folds=3, finalists=3),
        seed=42,
    )

    result = lab.automl.start(request).wait(timeout=300)

    # 1. O agente de dados preparou o Titanic, com a razão de cada decisão.
    titles = [decision.title for decision in result.plan.decisions]
    assert titles[:4] == ["Remover PassengerId", "Remover Name", "Remover Ticket", "Remover Cabin"]
    assert result.plan.objective.name == "accuracy"
    # 2. A triagem passou por todos os modelos de todas as famílias, com os valores por omissão.
    screening = [c for c in result.candidates if c.phase is SearchPhase.SCREENING]
    assert len(screening) == 9
    assert len({c.family for c in screening}) == 7
    assert all(not c.changes for c in screening)
    # 3. Só os três finalistas da triagem foram afinados.
    tuning = [c for c in result.candidates if c.phase is SearchPhase.TUNING]
    finalists = sorted(screening, key=lambda c: (-(c.mean or 0.0), c.std or 0.0, c.number))[:3]
    assert len(tuning) == 6
    assert {c.model for c in tuning} <= {c.model for c in finalists}
    # 4. O teste usou-se uma só vez: o registo tem um único run, o do vencedor.
    winner, run_id = result.winner_candidate, result.run_id
    assert winner is not None
    assert run_id is not None
    assert [entry.run_id for entry in lab.history.runs()] == [run_id]
    assert result.test is not None
    assert result.test.value is not None
    assert result.test.value > 0.75
    # 5. Na validação cruzada, o vencedor bate (ou empata) a referência.
    assert result.baseline is not None
    baseline = result.candidate(result.baseline)
    assert baseline.model == "logistic_regression"
    assert winner.mean is not None
    assert baseline.mean is not None
    assert winner.mean >= baseline.mean
    # 6. Fechar e reabrir a app: o vencedor está no disco e abre-se com o mesmo teste.
    after = Lab.create(fresh)
    entry = after.history.entry(run_id)
    assert entry.model_name == winner.model
    assert entry.name.startswith("Agentes · ")
    reopened = after.history.open(run_id)
    assert reopened.main_report.metric("accuracy").value == result.test.value


def test_the_data_screen_offers_the_two_modes(fresh: Settings) -> None:
    lab = Lab.create(fresh)
    summary = lab.datasets.load_sample("titanic")
    app = AppTest.from_file(APP, default_timeout=60)
    app.session_state["lab"] = lab
    app.session_state["dataset_id"] = summary.id
    app.run()

    assert not app.exception
    texts = [markdown.value for markdown in app.markdown]
    assert "**Modo automático**" in texts
    assert "**Modo manual**" in texts


def test_the_agents_screen_runs_a_search_and_opens_the_winner(fresh: Settings) -> None:
    lab = Lab.create(fresh)
    summary = lab.datasets.load_sample("iris")
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state["lab"] = lab
    app.session_state["dataset_id"] = summary.id
    app.run()
    app.switch_page("pages/agentes.py").run()
    assert not app.exception

    families = [ModelFamily.LINEAR, ModelFamily.TREE]
    app.multiselect(key="agentes.families.multiclass_classification").set_value(families).run()
    app.slider(key="agentes.settings.budget").set_value(4).run()
    app.slider(key="agentes.settings.folds").set_value(3).run()
    app.button(key="agentes.start").click().run()
    app.session_state["automl_job"].wait(timeout=120)
    app.run()  # o ecrã volta a correr e mostra o resultado

    assert not app.exception
    headers = [header.value for header in app.subheader]
    assert headers[:3] == ["A procura", "O melhor de cada agente", "O quadro dos candidatos"]
    assert len(app.get("plotly_chart")) == 2
    result = app.session_state["automl_result"]
    assert len(result.candidates) == 4

    app.button(key="agentes.open").click().run()

    assert not app.exception
    assert app.title[0].value == "Avaliar"
    assert app.session_state["result"].run_id == result.run_id


def test_a_candidate_is_trained_by_hand_from_the_board(fresh: Settings) -> None:
    lab = Lab.create(fresh)
    summary = lab.datasets.load_sample("iris")
    request = AutoMLRequest(
        dataset=summary.source,
        target="species",
        families=(ModelFamily.LINEAR, ModelFamily.TREE),
        settings=SearchSettings(budget=3, folds=3),
        seed=1,
    )
    result = lab.automl.run(request)
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state["lab"] = lab
    app.session_state["dataset_id"] = summary.id
    app.session_state["automl_result"] = result
    app.run()
    app.switch_page("pages/agentes.py").run()

    last = result.ranking[-1]
    app.selectbox(key=f"agentes.manual.{result.search_id}").set_value(last).run()
    app.button(key="agentes.train").click().run()

    assert not app.exception
    assert app.title[0].value == "Treinar"
    job = app.session_state["job"]
    trained = job.wait(timeout=60)
    assert trained.model_name == result.candidate(last).model
