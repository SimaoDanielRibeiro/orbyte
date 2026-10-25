"""Critério da fase 8: um agente aprende a atravessar a grelha e o run fica registado.

Pelos serviços: o Q-learning aprende o labirinto (o mapa mais difícil) e o DQN a grelha
simples; com a política aprendida, chegam sempre ao objetivo. O run fica no registo em
SQLite, e outro ``Lab`` com a mesma base de dados (como reabrir a app) encontra-o e abre-o.
Na interface: o ecrã Reinforcement learning mostra o que o agente aprendeu, e o ecrã
Experiências abre lá um run de um agente.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml_lab.config import Settings, get_settings
from ml_lab.core.services import (
    AgentSpec,
    EnvironmentSpec,
    Lab,
    Partition,
    ReinforcementConfig,
    TaskType,
)

APP = Path(__file__).resolve().parents[2] / "ui" / "simple" / "app.py"


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Uma base de dados e uma pasta de artefactos vazias, só deste teste (também para a app)."""
    monkeypatch.setenv("ML_LAB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ML_LAB_DATABASE_PATH", str(tmp_path / "artifacts" / "ml_lab.db"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def training(agent: str, layout: str, **hyperparameters: object) -> ReinforcementConfig:
    return ReinforcementConfig(
        environment=EnvironmentSpec(parameters={"layout": layout}),
        agent=AgentSpec(name=agent, hyperparameters=dict(hyperparameters)),
        seed=0,
    )


def test_an_agent_learns_to_cross_the_maze_and_the_run_is_recorded(fresh: Settings) -> None:
    before = Lab.create(fresh)

    result = before.reinforcement.run(training("q_learning", "labirinto"))

    assert result.reports[Partition.TEST].metric("success_rate").value == 1.0
    assert result.learned.reached
    assert len(result.learned.path) - 1 == 16  # o caminho mais curto do labirinto
    after = Lab.create(fresh)  # fechar e reabrir a app: só fica o que está no disco
    entry = after.history.entry(result.run_id)
    assert entry.task_type is TaskType.CONTROL
    assert (entry.model_name, entry.primary_value) == ("q_learning", 1.0)
    assert after.history.open_reinforcement(result.run_id).learned == result.learned


def test_the_deep_q_network_learns_the_grid_too(fresh: Settings) -> None:
    lab = Lab.create(fresh)

    result = lab.reinforcement.run(training("dqn", "simples", episodes=150))

    assert result.reports[Partition.TEST].metric("success_rate").value == 1.0
    assert lab.history.entry(result.run_id).model_name == "dqn"


def test_the_screen_shows_what_the_agent_learned(fresh: Settings) -> None:
    lab = Lab.create(fresh)
    result = lab.reinforcement.run(training("q_learning", "penhasco"))
    app = AppTest.from_file(APP, default_timeout=120)
    # O AppTest perde a barra lateral quando a app volta a correr: o estado vai antes.
    app.session_state["lab"] = lab
    app.session_state["rl_result"] = result
    app.run()

    app.switch_page("pages/reinforcement.py").run()

    assert not app.exception
    headers = [header.value for header in app.subheader]
    assert "O que o agente aprendeu" in headers
    assert len(app.get("plotly_chart")) == 3  # a recompensa, a grelha e a tabela Q
    first = result.trace.frames[0].episode if result.trace else 0
    app.select_slider(key=f"rl.episode.{result.run_id}").set_value(first).run()
    assert not app.exception


def test_the_history_opens_an_agent_on_its_screen(fresh: Settings) -> None:
    lab = Lab.create(fresh)
    result = lab.reinforcement.run(training("q_learning", "simples", episodes=200))
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state["lab"] = lab
    app.run()

    app.switch_page("pages/experiencias.py").run()
    app.selectbox(key="experiencias.run").set_value(result.run_id).run()
    app.button(key="experiencias.open").click().run()

    assert not app.exception
    assert app.session_state["rl_result"].run_id == result.run_id
