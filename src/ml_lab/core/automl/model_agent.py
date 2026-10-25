"""Os agentes de modelo: um por família, cada um responsável pelos modelos da sua família.

Os agentes não se escrevem à mão: saem do registo ``MODELS``. Cada família com modelos para
a tarefa tem um agente, e um modelo novo entra no agente da sua família sem mudar nada
aqui. O que cada agente sabe de cada modelo (o que vale a pena procurar) vem do próprio
modelo, no ``x-search`` dos hiperparâmetros (``search_space.py``).

Cada agente:

- na **triagem**, propõe os seus modelos com os valores por omissão: é a primeira
  impressão de cada um, barata;
- na **afinação**, propõe combinações novas para um modelo seu, sorteadas do espaço de
  procura, sem repetir nenhuma que já tenha experimentado;
- diz, em português, o que propôs e o que encontrou.

Ficam de fora, com a razão: as implementações didáticas que têm uma versão das bibliotecas
na família (dão o mesmo modelo, mais devagar; o modo manual tem todas) e, quando o objetivo
mede probabilidades, os modelos que não as dão.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Final

import numpy as np

from ml_lab.core.automl.objective import Objective
from ml_lab.core.automl.search_log import Candidate, model_agent_name
from ml_lab.core.automl.search_space import SearchSpace, search_space, written_value
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, Implementation, LabModel, ModelFamily
from ml_lab.formatting import format_list

ATTEMPTS: Final = 50
"""Quantas combinações um agente sorteia, no máximo, à procura de uma que ainda não experimentou."""


@dataclass(frozen=True, slots=True)
class SkippedModel:
    """Um modelo da tarefa que nenhum agente experimenta, e porquê.

    Attributes:
        model: O modelo, no registo.
        display_name: O nome a mostrar.
        family: A família.
        reason: Porque fica de fora.
    """

    model: str
    display_name: str
    family: ModelFamily
    reason: str


def describe_skipped(skipped: Sequence[SkippedModel]) -> str:
    """Os modelos que ficam de fora, agrupados pela razão ("A e B: <razão>; C: <razão>")."""
    groups: dict[str, list[str]] = {}
    for model in skipped:
        groups.setdefault(model.reason, []).append(model.display_name)
    return "; ".join(f"{format_list(names)}: {reason}" for reason, names in groups.items())


class ModelAgent:
    """O agente de uma família de modelos.

    Args:
        family: A família.
        models: Os modelos da família que o agente experimenta (pela ordem do catálogo).
    """

    def __init__(self, family: ModelFamily, models: Sequence[type[LabModel[Any]]]) -> None:
        self.family = family
        self.models = tuple(models)
        self._spaces = {model.info.name: search_space(model) for model in self.models}
        self._tried: set[str] = set()

    @property
    def name(self) -> str:
        """O nome a mostrar ("Agente · Árvores de decisão")."""
        return model_agent_name(self.family)

    def space(self, model: type[LabModel[Any]]) -> SearchSpace:
        """O espaço de procura de um modelo do agente."""
        return self._spaces[model.info.name]

    def defaults(self, model: type[LabModel[Any]]) -> dict[str, Any]:
        """Os valores por omissão de um modelo, como na triagem."""
        return model.info.hyperparameters().model_dump(mode="json")

    def remember(self, model: type[LabModel[Any]], values: dict[str, Any]) -> None:
        """Toma nota de uma combinação experimentada (para nunca a repetir)."""
        self._tried.add(_key(model, values))

    def propose(
        self, model: type[LabModel[Any]], rng: np.random.Generator
    ) -> dict[str, Any] | None:
        """Uma combinação nova para ``model``: os valores por omissão, com os do espaço sorteados.

        Returns:
            Os hiperparâmetros completos, ou ``None`` se o modelo não tiver nada para procurar
            ou o agente não encontrar uma combinação nova (já experimentou o espaço todo).
        """
        space = self.space(model)
        if space.is_empty:
            return None
        defaults = self.defaults(model)
        for _ in range(ATTEMPTS):
            values = {**defaults, **space.sample(rng)}
            if _key(model, values) not in self._tried:
                return values
        return None

    def report(self, candidate: Candidate, best_before: float | None, objective: Objective) -> str:
        """O que o agente diz depois de medir um candidato seu.

        Args:
            candidate: O candidato medido.
            best_before: Na afinação, o melhor resultado do mesmo modelo até aí.
            objective: O objetivo (para escrever os valores).
        """
        name = candidate.model_display_name
        changed = describe_changes(candidate.changes)
        how = f"com {changed}" if changed else "com os valores por omissão"
        if candidate.failed:
            return f"{name} {how}: falhou ({candidate.error})"
        score = f"{objective.written(candidate.mean)} ± {objective.written(candidate.std)}"
        if best_before is None:
            return f"{name} {how}: {score}."
        before = objective.written(best_before)
        if objective.is_better(candidate.mean, best_before):
            return f"{name} {how}: {score}, melhor do que os {before} que já tinha."
        return f"{name} {how}: {score}, abaixo dos {before} que já tinha."


def model_agents(
    task_type: TaskType,
    objective: Objective,
    *,
    families: Sequence[ModelFamily] = (),
    include_didactic: bool = False,
) -> tuple[tuple[ModelAgent, ...], tuple[SkippedModel, ...]]:
    """Os agentes de modelo para ``task_type``, e os modelos que ficam de fora (com a razão).

    Args:
        task_type: A tarefa.
        objective: O objetivo (se medir probabilidades, só entram os modelos que as dão).
        families: As famílias que participam; vazio = todas.
        include_didactic: Se entram também as implementações didáticas.

    Raises:
        InvalidParameterError: Se nenhum modelo puder participar.
    """
    found = sorted(
        MODELS.for_task(task_type).values(),
        key=lambda model: (model.info.family.order, model.info.display_name),
    )
    with_library = {
        model.info.family
        for model in found
        if model.info.implementation is not Implementation.DIDACTIC
    }
    kept: list[type[LabModel[Any]]] = []
    skipped: list[SkippedModel] = []
    # As razões servem para um modelo e para uma lista deles ("A, B e C: <razão>").
    for model in found:
        info = model.info
        reason = None
        if families and info.family not in families:
            reason = "família que não foi escolhida para esta procura"
        elif not info.supports_new_data:
            reason = "só conhece as linhas com que treinou, não se avalia em dados novos"
        elif objective.needs_proba and not info.supports_proba:
            reason = f"sem probabilidades, de que {objective.label} precisa"
        elif (
            info.implementation is Implementation.DIDACTIC
            and not include_didactic
            and info.family in with_library
        ):
            reason = (
                "implementação didática, feita para se ver o treino por dentro: os agentes usam "
                "a das bibliotecas, mais rápida (o modo manual tem todas)"
            )
        if reason is None:
            kept.append(model)
        else:
            skipped.append(SkippedModel(info.name, info.display_name, info.family, reason))
    if not kept:
        raise InvalidParameterError(
            f"Nenhum modelo de {task_type.label.lower()} pode entrar nesta procura: escolhe "
            "outras famílias, ou outra métrica."
        )
    agents = tuple(
        ModelAgent(family, list(members))
        for family, members in groupby(kept, key=lambda model: model.info.family)
    )
    return agents, tuple(skipped)


def describe_changes(changes: dict[str, Any]) -> str:
    """Os hiperparâmetros mudados, em texto ("max_depth 8, min_samples_leaf 5")."""
    return ", ".join(f"{name} {written_value(value)}" for name, value in changes.items())


def _key(model: type[LabModel[Any]], values: dict[str, Any]) -> str:
    return json.dumps({"model": model.info.name, **values}, sort_keys=True, default=str)
