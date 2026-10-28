"""Contratos do reinforcement learning: aplicados a TODOS os agentes e ambientes registados.

Um agente ou um ambiente novo fica testado aqui sem escrever testes novos, como os modelos
em ``test_models.py``.
"""

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_lab.core.foundation.errors import NotFittedError
from ml_lab.core.rl.agents.base import AGENTS, Agent, greedy_path
from ml_lab.core.rl.environments.base import ENVIRONMENTS, CellKind, Environment
from ml_lab.core.training.context import RecordingTrainingContext
from ml_lab.core.training.step_metrics import STEP_METRICS

DOCS = Path(__file__).resolve().parents[2] / "docs" / "algorithms"
TEMPLATE_SECTIONS = (
    "## Intuição",
    "## Matemática essencial",
    "## Hiperparâmetros",
    "## Armadilhas",
)
EPISODES = 150

AGENT_NAMES = AGENTS.names()
ENVIRONMENT_NAMES = ENVIRONMENTS.names()


def simple_grid() -> Environment[Any]:
    return ENVIRONMENTS.get("gridworld")()


def agent(name: str, seed: int = 0, **values: Any) -> Agent[Any]:
    agent_class = AGENTS.get(name)
    parameters = agent_class.info.hyperparameters.parse({"episodes": EPISODES, **values})
    return agent_class(parameters, seed=seed)


def test_there_are_agents_and_environments() -> None:
    assert AGENT_NAMES
    assert ENVIRONMENT_NAMES


# --- agentes ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_every_agent_learns_to_cross_the_simple_grid(name: str) -> None:
    environment = simple_grid()

    trained = agent(name).train(environment)

    path, reached = greedy_path(trained.q_values(), environment)
    assert reached
    assert len(path) - 1 == 8  # o caminho mais curto da grelha simples


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_the_same_seed_gives_the_same_training(name: str) -> None:
    first = agent(name, seed=3).train(simple_grid())
    second = agent(name, seed=3).train(simple_grid())

    assert np.array_equal(first.q_values(), second.q_values())


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_each_episode_is_a_step_of_the_training(name: str) -> None:
    context = RecordingTrainingContext()

    agent(name).train(simple_grid(), context)

    assert [step for step, _, _ in context.steps] == list(range(1, EPISODES + 1))
    _, loss, metrics = context.steps[-1]
    assert loss is not None  # aprendeu alguma coisa no último episódio
    assert loss >= 0
    assert set(metrics) == {"reward", "length", "success", "epsilon"}
    # Cada métrica do episódio tem nome a mostrar no vocabulário (os gráficos do treino).
    assert set(metrics) <= STEP_METRICS.keys()
    epsilons = [m["epsilon"] for _, _, m in context.steps]
    assert epsilons == sorted(epsilons, reverse=True)  # a exploração só desce


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_every_agent_stops_when_asked(name: str) -> None:
    context = RecordingTrainingContext(stop_after=5)

    agent(name).train(simple_grid(), context)

    assert len(context.steps) == 5


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_snapshots_carry_the_q_table_and_the_path(name: str) -> None:
    environment = simple_grid()
    context = RecordingTrainingContext()

    agent(name).train(environment, context)

    _, state = context.snapshots[-1]
    assert np.asarray(state["q_table"]).shape == (environment.n_states, environment.n_actions)
    assert state["path"][0] == environment.reset(np.random.default_rng(0))
    assert isinstance(state["reached"], bool)


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_misuse_is_explained_and_the_state_is_plain_json(name: str) -> None:
    fresh = agent(name)

    with pytest.raises(NotFittedError):
        fresh.act(0)
    assert fresh.state() == {}
    trained = fresh.train(simple_grid())
    json.dumps(trained.state())


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_every_agent_has_a_didactic_card_with_its_hyperparameters(name: str) -> None:
    info = AGENTS.get(name).info
    card = DOCS / info.doc

    assert card.is_file(), f"falta a ficha didática {card}"
    text = card.read_text(encoding="utf-8")
    assert all(section in text for section in TEMPLATE_SECTIONS)
    for field in info.hyperparameters.model_fields:
        assert f"`{field}`" in text, f"a ficha não explica o hiperparâmetro {field}"


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_the_agent_card_matches_the_registry(name: str) -> None:
    info = AGENTS.get(name).info
    defaults = info.hyperparameters()
    schema = info.hyperparameters.form_schema()

    assert info.name == name
    assert info.display_name
    assert info.summary
    assert info.implementation.label
    assert info.step_limit in info.hyperparameters.model_fields
    assert info.hyperparameters.parse(defaults.model_dump(mode="json")) == defaults
    for field, description in schema["properties"].items():
        assert description.get("title"), f"{field} precisa de um título para o formulário"
        assert description.get("description"), f"{field} precisa de uma explicação"


# --- ambientes --------------------------------------------------------------------------


def every_map(name: str) -> list[Environment[Any]]:
    """O ambiente com cada opção dos seus parâmetros de escolha (os mapas, na grelha)."""
    environment_class = ENVIRONMENTS.get(name)
    parameters = environment_class.info.parameters
    choices = {
        field: list(spec["enum"])
        for field, spec in parameters.form_schema()["properties"].items()
        if "enum" in spec
    }
    if not choices:
        return [environment_class()]
    field, options = next(iter(choices.items()))
    return [environment_class(parameters.parse({field: option})) for option in options]


ENVIRONMENTS_AND_MAPS = [
    pytest.param(environment, id=f"{name}-{environment.title()}")
    for name in ENVIRONMENT_NAMES
    for environment in every_map(name)
]


@pytest.mark.parametrize("environment", ENVIRONMENTS_AND_MAPS)
def test_the_goal_can_be_reached_from_the_start(environment: Environment[Any]) -> None:
    start = environment.reset(np.random.default_rng(0))
    seen, queue = {start}, deque([start])
    while queue:
        state = queue.popleft()
        for action in range(environment.n_actions):
            transition = environment.step(state, action, None)
            if transition.reached_goal:
                return
            if not transition.done and transition.state not in seen:
                seen.add(transition.state)
                queue.append(transition.state)
    pytest.fail(f"o objetivo de {environment.title()} não se alcança a partir da partida")


@pytest.mark.parametrize("environment", ENVIRONMENTS_AND_MAPS)
def test_every_step_stays_inside_and_never_enters_a_wall(environment: Environment[Any]) -> None:
    layout = environment.layout()

    for state, kind in enumerate(layout.cells):
        if kind is CellKind.WALL:
            continue
        for action in range(environment.n_actions):
            transition = environment.step(state, action, None)
            assert 0 <= transition.state < environment.n_states
            assert layout.kind(transition.state) is not CellKind.WALL
            assert transition.done == (
                layout.kind(transition.state) in {CellKind.GOAL, CellKind.TRAP}
            )


@pytest.mark.parametrize("environment", ENVIRONMENTS_AND_MAPS)
def test_the_layout_describes_the_environment(environment: Environment[Any]) -> None:
    layout = environment.layout()

    assert len(layout.cells) == layout.rows * layout.columns == environment.n_states
    assert len(layout.action_labels) == len(layout.action_moves) == environment.n_actions
    assert layout.cells.count(CellKind.START) == 1
    assert CellKind.GOAL in layout.cells
    assert environment.max_steps > 0
