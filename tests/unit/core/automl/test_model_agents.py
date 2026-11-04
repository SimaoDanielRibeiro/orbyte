"""Os agentes de modelo: um por família, saídos do registo, e o que cada um propõe."""

from itertools import product
from typing import Any

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
    assert ensembles.name == "Agent · Ensembles"


def test_every_model_of_the_task_joins_by_default() -> None:
    agents, skipped = model_agents(BINARY, objective_for(BINARY))

    joined = {model.info.name for agent in agents for model in agent.models}
    assert joined == set(MODELS.for_task(BINARY))
    assert skipped == ()
    # As implementações didáticas também: antes ficavam de fora (ADR 0049).
    linear = next(agent for agent in agents if agent.family is ModelFamily.LINEAR)
    assert {model.info.name for model in linear.models} == {
        "logistic_regression",
        "logistic_regression_gd",
        "perceptron",
    }


def test_only_the_chosen_models_take_part() -> None:
    agents, skipped = model_agents(
        BINARY, objective_for(BINARY), models=["decision_tree", "knn_numpy"]
    )

    assert [(agent.family, [model.info.name for model in agent.models]) for agent in agents] == [
        (ModelFamily.INSTANCE_BASED, ["knn_numpy"]),
        (ModelFamily.TREE, ["decision_tree"]),
    ]
    assert all(model.chosen_out for model in skipped)
    assert {model.reason for model in skipped} == {"not chosen for this search"}
    # Todos com a mesma razão vão juntos numa só frase.
    assert describe_skipped(skipped).count("not chosen") == 1


def test_a_model_of_another_task_is_refused() -> None:
    with pytest.raises(InvalidParameterError, match="can't search kmeans"):
        model_agents(BINARY, objective_for(BINARY), models=["kmeans"])


def test_an_objective_with_probabilities_leaves_out_the_models_without_them() -> None:
    _, skipped = model_agents(BINARY, objective_for(BINARY, "roc_auc"))

    assert [(model.model, model.reason, model.chosen_out) for model in skipped] == [
        ("perceptron", "no probabilities, which ROC AUC needs", False)
    ]


def test_a_model_that_cant_be_measured_stays_out_even_if_chosen() -> None:
    with pytest.raises(InvalidParameterError, match="No binary classification model"):
        model_agents(BINARY, objective_for(BINARY, "roc_auc"), models=["perceptron"])


def test_only_the_chosen_families_take_part() -> None:
    agents, skipped = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])

    assert [agent.family for agent in agents] == [ModelFamily.TREE]
    assert {model.reason for model in skipped} == {"not chosen for this search"}


def test_a_search_without_any_model_is_refused() -> None:
    with pytest.raises(InvalidParameterError, match="model can join this search"):
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


def test_the_agent_reports_what_it_found_in_plain_words() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])
    accuracy = objective_for(BINARY)
    tuned = Candidate(
        number=5,
        phase=SearchPhase.TUNING,
        family=ModelFamily.TREE,
        model="decision_tree",
        model_display_name="Decision tree",
        hyperparameters={"max_depth": 3, "min_samples_leaf": 5, "criterion": "gini"},
        changes={"max_depth": 3, "min_samples_leaf": 5},
        mean=0.812,
        std=0.021,
    )

    text = agents[0].report(tuned, 0.79, accuracy, leader_before=0.83)

    # Os nomes que se mostram, o valor por omissão ao lado do novo, e o resultado lido.
    assert text == (
        "I tried Decision tree with maximum depth 3 (default 5) and minimum examples per leaf 5 "
        "(default 1). Accuracy in the cross-validation: 81.2% ± 2.1% across the folds. It gets "
        "about 81 in every 100 rows right. That beats the 79.0% it had: these settings help."
    )


def screened(**values: object) -> Candidate:
    """Um candidato da triagem (com os valores por omissão) da árvore de decisão."""
    fields: dict[str, Any] = {
        "number": 1,
        "phase": SearchPhase.SCREENING,
        "family": ModelFamily.TREE,
        "model": "decision_tree",
        "model_display_name": "Decision tree",
        "hyperparameters": {},
        "changes": {},
        "mean": 0.8,
        "std": 0.02,
    }
    return Candidate(**{**fields, **values})


def test_the_first_result_is_the_mark_to_beat_and_a_new_leader_is_said() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])
    accuracy = objective_for(BINARY)

    first = agents[0].report(screened(), None, accuracy, first=True)
    leader = agents[0].report(screened(mean=0.85), None, accuracy, leader_before=0.8)

    assert first.startswith("I measured Decision tree with its default settings.")
    assert first.endswith("It's the first result of the search: the mark the others must beat.")
    assert leader.endswith("It's now the best of the whole search.")


def test_the_agent_says_when_a_model_memorises_the_training_data() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])
    accuracy = objective_for(BINARY)

    memorises = agents[0].report(screened(train_mean=0.99), None, accuracy, leader_before=0.9)
    fine = agents[0].report(screened(train_mean=0.84), None, accuracy, leader_before=0.9)

    assert "On the rows it trained on it scored 99.0%" in memorises
    assert "memorises" not in fine


def test_the_special_values_are_written_by_name() -> None:
    agents, _ = model_agents(BINARY, objective_for(BINARY), families=[ModelFamily.TREE])
    tree = MODELS.get("decision_tree")

    assert agents[0].explain_changes(tree, {"max_depth": 0}) == "maximum depth no limit (default 5)"
