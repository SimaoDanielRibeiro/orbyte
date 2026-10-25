"""O agente de dados: o que decide para o Titanic, e porquê, a partir do perfil."""

from typing import Any

import pandas as pd
import pytest

from ml_lab.core.automl.data_agent import DataAgent, DataPlan
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.profiling.dataset_profile import profile_dataset
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.task.task_detection import suggest_task
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.services.datasets import DatasetService

TITANIC = SourceSpec("sample", {"name": "titanic"})


def titanic(**kwargs: Any) -> DataPlan:
    datasets = DatasetService()
    summary = datasets.load(TITANIC)
    return DataAgent().plan(
        datasets.profile(summary.id, "Survived"),
        datasets.suggest_task(summary.id, "Survived"),
        target="Survived",
        **kwargs,
    )


def from_frame(frame: pd.DataFrame, target: str) -> DataPlan:
    return DataAgent().plan(
        profile_dataset(frame, target=target), suggest_task(frame, target), target=target
    )


def test_the_data_agent_cleans_the_titanic_with_a_reason_for_each_step() -> None:
    plan = titanic()

    titles = [decision.title for decision in plan.decisions]
    assert titles[:4] == ["Remover PassengerId", "Remover Name", "Remover Ticket", "Remover Cabin"]
    assert "identificador" in plan.decisions[0].reason
    assert "texto livre" in plan.decisions[1].reason
    columns = [step.params["columns"] for step in plan.preparation.cleaning]
    assert columns == [["PassengerId"], ["Name"], ["Ticket"], ["Cabin"]]


def test_the_task_is_confirmed_and_the_split_keeps_the_test_aside() -> None:
    plan = titanic()

    assert plan.task_type is TaskType.BINARY_CLASSIFICATION
    assert plan.preparation.split == SplitConfig(SplitStrategy.STRATIFIED, test_size=0.2)
    split = next(d for d in plan.decisions if d.title.startswith("Divisão"))
    assert split.title == "Divisão estratificada: 80% para treino, 20% para teste"
    assert "só o vencedor é avaliado no teste" in split.reason


def test_the_transforms_are_explained_from_the_profile() -> None:
    titles = [decision.title for decision in titanic().decisions]

    assert "Preencher os nulos de Age com a mediana" in titles
    assert "Preencher os nulos de Embarked com o valor mais frequente" in titles
    assert "One-hot em Sex e Embarked" in titles
    assert "Padronizar as numéricas (média 0, desvio padrão 1)" in titles


def test_balanced_classes_keep_the_accuracy() -> None:
    plan = titanic()

    assert plan.objective == "accuracy"
    assert "38,4%" in plan.decisions[-1].reason  # a classe mais rara do Titanic


def test_imbalanced_classes_switch_to_the_balanced_accuracy() -> None:
    frame = pd.DataFrame({"x": [float(i) for i in range(40)], "fraud": ["no"] * 36 + ["yes"] * 4})

    plan = from_frame(frame, "fraud")

    assert plan.objective == "balanced_accuracy"
    assert "desequilibradas" in plan.decisions[-1].reason


def test_a_requested_objective_is_respected_and_the_suggestion_is_still_told() -> None:
    plan = titanic(objective="f1")

    assert plan.objective == "f1"
    assert plan.decisions[-1].reason.startswith(
        "Foi a métrica pedida (o agente de dados sugeria Exatidão). As classes estão"
    )


def test_asking_for_the_suggested_objective_keeps_the_agents_reason() -> None:
    # Um formulário manda sempre a métrica escolhida, mesmo que seja a sugerida.
    assert titanic(objective="accuracy").decisions == titanic().decisions


def test_a_requested_task_must_fit_the_target() -> None:
    with pytest.raises(InvalidParameterError, match="não serve para o alvo Survived"):
        titanic(task_type=TaskType.REGRESSION)


def test_a_preparation_made_by_hand_is_adopted_as_it_is() -> None:
    own = PreparationConfig(target="Survived", split=SplitConfig(SplitStrategy.RANDOM, 0.3))

    plan = titanic(preparation=own)

    assert plan.adopted
    assert plan.preparation.split.test_size == 0.3
    assert plan.decisions[0].title == "Usar a preparação do ecrã Preparar"


def test_a_preparation_for_another_target_is_refused() -> None:
    with pytest.raises(InvalidParameterError, match="alvo Age"):
        titanic(preparation=PreparationConfig(target="Age"))


def test_a_target_with_a_single_value_cannot_be_predicted() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": ["a", "a", "a", "a"]})

    with pytest.raises(DataError, match="sempre o mesmo valor"):
        from_frame(frame, "y")
