"""Os agentes de modelo: um por família, saídos do registo, e o que cada um propõe."""

from itertools import product

import numpy as np
import pytest

from ml_lab.core.automl.model_agent import describe_skipped, model_agents
from ml_lab.core.automl.objective import objective_for
from ml_lab.core.automl.search_log import Candidate, SearchPhase
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, ModelFamily

BINARY = TaskType.BINARY_CLASSIFICATION


def test_there_is_one_agent_per_family_with_the_models_of_that_family() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY))

    assert [agent.family for agent in agents] == [
        ModelFamily.LINEAR,
        ModelFamily.INSTANCE_BASED,
        ModelFamily.PROBABILISTIC,
        ModelFamily.TREE,
        ModelFamily.ENSEMBLE,
        ModelFamily.KERNEL,
        ModelFamily.NEURAL,
    ]
    ensembles = next(agent for agent in agents if agent.family is ModelFamily.ENSEMBLE)
    assert [model.info.name for model in ensembles.models] == [
        "adaboost",
        "gradient_boosting",
        "random_forest",
    ]
    assert ensembles.name == "Agente · Ensembles (conjuntos de modelos)"


def test_the_didactic_twins_stay_out_with_a_reason() -> None:
    _, skipped = model_agents(BINARY, objective_for(BINARY))

    assert {model.model for model in skipped} == {
        "knn_numpy",
        "logistic_regression_gd",
        "perceptron",
    }
    assert all("didática" in model.reason for model in skipped)
    # Os três com a mesma razão vão juntos numa só frase.
    assert describe_skipped(skipped).count("didática") == 1


def test_the_didactic_models_can_be_called_in() -> None:
    agents, skipped = model_agents(BINARY, objective_for(BINARY), include_didactic=True)

    linear = next(agent for agent in agents if agent.family is ModelFamily.LINEAR)
    assert {model.info.name for model in linear.models} == {
        "logistic_regression",
        "logistic_regression_gd",
        "perceptron",
    }
    assert skipped == ()


def test_an_objective_with_probabilities_leaves_out_the_models_without_them() -> None:
    _, skipped = model_agents(BINARY, objective_for(BINARY, "roc_auc"), include_didactic=True)

    assert [(model.model, model.reason) for model in skipped] == [
        ("perceptron", "sem probabilidades, de que ROC AUC precisa")
    ]


def test_only_the_chosen_families_take_part() -> None:
    agents, skipped = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])

    assert [agent.family for agent in agents] == [ModelFamily.TREE]
    assert {model.reason for model in skipped} == {
        "família que não foi escolhida para esta procura"
    }


def test_a_search_without_any_model_is_refused() -> None:
    with pytest.raises(InvalidParameterError, match="Nenhum modelo"):
        model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.CENTROID])


def test_a_proposal_is_valid_and_never_repeats_a_combination() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])
    agent = agents[0]
    tree = MODELS.get("decision_tree")
    rng = np.random.default_rng(0)
    agent.remember(tree, agent.defaults(tree))

    proposals = []
    for _ in range(20):
        values = agent.propose(tree, rng)
        assert values is not None
        tree.info.hyperparameters.parse(values)  # válida: está dentro dos limites do modelo
        agent.remember(tree, values)
        proposals.append(values)

    keys = [tuple(sorted(values.items())) for values in proposals]
    assert len(set(keys)) == 20
    assert agent.defaults(tree) not in proposals


def test_an_agent_knows_when_it_has_tried_every_combination() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])
    agent = agents[0]
    tree = MODELS.get("decision_tree")
    space = agent.space(tree)
    defaults = agent.defaults(tree)
    for combination in product(*(dimension.values for dimension in space.dimensions)):
        names = [dimension.name for dimension in space.dimensions]
        agent.remember(tree, {**defaults, **dict(zip(names, combination, strict=True))})

    assert agent.propose(tree, np.random.default_rng(0)) is None


def test_a_model_with_nothing_to_tune_has_no_proposal() -> None:
    regression = TaskType.REGRESSION
    agents, _ = model_agents(regression, objective_for(regression), families=[ModelFamily.LINEAR])
    linear = MODELS.get("linear_regression")

    assert agents[0].propose(linear, np.random.default_rng(0)) is None


def test_the_agent_reports_what_it_found_in_portuguese() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])
    accuracy = objective_for(BINARY)
    tuned = Candidate(
        number=5,
        phase=SearchPhase.TUNING,
        family=ModelFamily.TREE,
        model="decision_tree",
        model_display_name="Árvore de decisão",
        hyperparameters={"max_depth": 3, "min_samples_leaf": 5, "criterion": "gini"},
        changes={"max_depth": 3, "min_samples_leaf": 5},
        mean=0.812,
        std=0.021,
    )

    text = agents[0].report(tuned, 0.79, accuracy)

    assert text == (
        "Árvore de decisão com max_depth 3, min_samples_leaf 5: 81,2% ± 2,1%, melhor do que os "
        "79,0% que já tinha."
    )
