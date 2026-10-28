"""Critério da fase 8: um agente aprende a atravessar a grelha e o run fica registado.

Pelos serviços: o Q-learning aprende o labirinto (o mapa mais difícil) e o DQN a grelha
simples; com a política aprendida, chegam sempre ao objetivo. O run fica no registo em
SQLite, e outro ``Lab`` com a mesma base de dados (como reabrir a app) encontra-o e abre-o.
Na interface, o ecrã do RL (o que o agente aprendeu, episódio a episódio) está nos testes
de ponta a ponta da web (``ui/web/e2e/reinforcement.spec.ts``).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.services import (
    AgentSpec,
    EnvironmentSpec,
    Lab,
    Partition,
    ReinforcementConfig,
    TaskType,
)


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Uma base de dados e uma pasta de artefactos vazias, só deste teste."""
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
