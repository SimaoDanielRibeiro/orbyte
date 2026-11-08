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
- diz, em inglês e por palavras, o que experimentou e o que encontrou (ADR 0049): os
  hiperparâmetros pelo nome que se mostra, com o valor por omissão ao lado do novo, o
  resultado lido ("It explains about 75% of the variation in the target"), se ajudou e se o
  modelo decora o treino.

Entram todos os modelos da tarefa (ADR 0049), menos os que não se podem medir: os que só
conhecem as linhas com que treinaram e, quando o objetivo mede probabilidades, os que não as
dão. Quem faz o pedido escolhe, além disso, os modelos (e as famílias) que entram.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Final

import numpy as np

from ml_lab.core.automl.objective import Objective
from ml_lab.core.automl.search_log import Candidate, model_agent_name
from ml_lab.core.automl.search_space import Dimension, SearchSpace, search_space, written_value
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, LabModel, ModelFamily
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
        chosen_out: Se ficou de fora por escolha de quem fez o pedido (e não porque não se
            pode medir): o ecrã Agentes deixa voltar a escolhê-lo.
    """

    model: str
    display_name: str
    family: ModelFamily
    reason: str
    chosen_out: bool = False


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
        self._by_name = {model.info.name: model for model in self.models}
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

    def has_tried(self, model: type[LabModel[Any]], values: dict[str, Any]) -> bool:
        """Se a combinação já foi experimentada (por este agente ou pelo agente de IA)."""
        return _key(model, values) in self._tried

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

    def explain_changes(self, model: type[LabModel[Any]], changes: Mapping[str, Any]) -> str:
        """Os hiperparâmetros mudados, por palavras, com o valor por omissão ao lado.

        "maximum depth 8 (default 5) and minimum examples per leaf 10 (default 1)": o nome que
        se mostra em vez do nome no código, e o ponto de partida, para se ver o que mudou.
        """
        return changes_in_words(model, changes, space=self.space(model))

    def report(
        self,
        candidate: Candidate,
        best_before: float | None,
        objective: Objective,
        *,
        leader_before: float | None = None,
        first: bool = False,
    ) -> str:
        """O que o agente diz depois de medir um candidato seu, por palavras (ADR 0049).

        Diz o que experimentou, o resultado e o que ele quer dizer, se ajudou (na afinação) e,
        quando o treino fica muito acima da validação, que o modelo decora o treino.

        Args:
            candidate: O candidato medido.
            best_before: Na afinação, o melhor resultado do mesmo modelo até aí.
            objective: O objetivo (para escrever e ler os valores).
            leader_before: O melhor resultado da procura antes deste candidato.
            first: Se é o primeiro candidato da procura (a marca que os outros têm de bater).
        """
        name = candidate.model_display_name
        model = self._by_name.get(candidate.model)
        if candidate.changes and model is not None:
            tried = f"I tried {name} with {self.explain_changes(model, candidate.changes)}."
        elif candidate.changes:
            tried = f"I tried {name} with {describe_changes(candidate.changes)}."
        else:
            tried = f"I measured {name} with its default settings."
        if candidate.failed:
            return f"{tried} It failed ({candidate.error}), so it doesn't count."
        score = (
            f"{objective.label} in the cross-validation: {objective.written(candidate.mean)} "
            f"± {objective.written(candidate.std)} across the folds."
        )
        sentences = [tried, score]
        reading = objective.reading(candidate.mean)
        if reading:
            sentences.append(f"{reading}.")
        # Comparam-se os valores como se mostram: 96.6% contra 96.6% é um empate para quem lê,
        # mesmo que um ganhe na quarta casa decimal.
        now = objective.written(candidate.mean)
        if best_before is not None:
            before = objective.written(best_before)
            if now == before:
                sentences.append(f"The same as the {before} it had.")
            elif objective.is_better(candidate.mean, best_before):
                sentences.append(f"That beats the {before} it had: these settings help.")
            else:
                sentences.append(
                    f"That's below the {before} it had: these settings don't help, and its best "
                    f"stays at {before}."
                )
        led = best_before is not None and objective.written(best_before) == objective.written(
            leader_before
        )
        if first:
            sentences.append("It's the first result of the search: the mark the others must beat.")
        elif leader_before is not None and now == objective.written(leader_before):
            if not led:
                sentences.append("It ties with the best of the search.")
        elif objective.is_better(candidate.mean, leader_before):
            sentences.append("It's now the best of the whole search.")
        train, validation = candidate.train_mean, candidate.mean
        if train is not None and validation is not None and objective.overfits(train, validation):
            sentences.append(
                f"On the rows it trained on it scored {objective.written(train)}, well above the "
                "folds it didn't see: it memorises part of the training data."
            )
        return " ".join(sentences)


def model_agents(
    task_type: TaskType,
    objective: Objective,
    *,
    families: Sequence[ModelFamily] = (),
    models: Sequence[str] = (),
) -> tuple[tuple[ModelAgent, ...], tuple[SkippedModel, ...]]:
    """Os agentes de modelo para ``task_type``, e os modelos que ficam de fora (com a razão).

    Entram todos os modelos da tarefa que se podem medir com ``objective``; ``families`` e
    ``models`` escolhem quais desses entram (vazios, entram todos).

    Args:
        task_type: A tarefa.
        objective: O objetivo (se medir probabilidades, só entram os modelos que as dão).
        families: As famílias que participam; vazio = todas.
        models: Os modelos que participam, pelo nome no registo; vazio = todos.

    Raises:
        InvalidParameterError: Se ``models`` tiver um modelo que não é desta tarefa, ou se
            nenhum modelo puder participar.
    """
    found = sorted(
        MODELS.for_task(task_type).values(),
        key=lambda model: (model.info.family.order, model.info.display_name),
    )
    known = {model.info.name for model in found}
    unknown = [name for name in models if name not in known]
    if unknown:
        what = "isn't a model" if len(unknown) == 1 else "aren't models"
        raise InvalidParameterError(
            f"The agents can't search {format_list(unknown)}: {what} for "
            f"{task_type.label.lower()}. Choose among the models listed for this task."
        )
    kept: list[type[LabModel[Any]]] = []
    skipped: list[SkippedModel] = []
    # As razões servem para um modelo e para uma lista deles ("A, B e C: <razão>"). As de quem
    # não se pode medir vêm primeiro: um modelo sem probabilidades não entra, escolhido ou não.
    for model in found:
        info = model.info
        reason = None
        chosen_out = False
        if not info.supports_new_data:
            reason = "it only knows the rows it trained on, it can't be evaluated on new data"
        elif objective.needs_proba and not info.supports_proba:
            reason = f"no probabilities, which {objective.label} needs"
        elif (families and info.family not in families) or (models and info.name not in models):
            reason, chosen_out = "not chosen for this search", True
        if reason is None:
            kept.append(model)
        else:
            skipped.append(
                SkippedModel(info.name, info.display_name, info.family, reason, chosen_out)
            )
    if not kept:
        raise InvalidParameterError(
            f"No {task_type.label.lower()} model can join this search: choose at least one of "
            "the models the agents offer, or another metric."
        )
    agents = tuple(
        ModelAgent(family, list(members))
        for family, members in groupby(kept, key=lambda model: model.info.family)
    )
    return agents, tuple(skipped)


def describe_changes(changes: dict[str, Any]) -> str:
    """Os hiperparâmetros mudados, em texto ("max_depth 8, min_samples_leaf 5")."""
    return ", ".join(f"{name} {written_value(value)}" for name, value in changes.items())


def changes_in_words(
    model: type[LabModel[Any]], changes: Mapping[str, Any], *, space: SearchSpace | None = None
) -> str:
    """Os hiperparâmetros mudados de ``model``, por palavras, com o valor por omissão ao lado.

    É o que ``ModelAgent.explain_changes`` diz, sem precisar do agente: o analista (ADR 0053)
    lê assim os candidatos de uma procura que já acabou. Sem mudanças, diz que ficaram os
    valores por omissão.

    Args:
        model: O modelo.
        changes: Os hiperparâmetros que mudaram (o ``Candidate.changes``).
        space: O espaço de procura do modelo, se já o houver; senão, lê-se do modelo.
    """
    if not changes:
        return "its default settings"
    found = space if space is not None else search_space(model)
    dimensions = {dimension.name: dimension for dimension in found.dimensions}
    defaults = model.info.hyperparameters().model_dump(mode="json")
    return format_list(
        [
            _change_in_words(dimensions.get(name), name, value, defaults.get(name))
            for name, value in changes.items()
        ]
    )


def _change_in_words(dimension: Dimension | None, name: str, value: Any, default: Any) -> str:
    """Um hiperparâmetro mudado: o nome que se mostra, o valor novo e o de omissão."""
    if dimension is None:
        return f"{name} {written_value(value)} (default {written_value(default)})"
    return (
        f"{_lowered(dimension.title)} {dimension.shown(value)} (default {dimension.shown(default)})"
    )


def _lowered(title: str) -> str:
    """O nome no meio de uma frase: "Maximum depth" passa a "maximum depth", mas "C" fica."""
    if len(title) > 1 and title[0].isupper() and title[1].islower():
        return title[0].lower() + title[1:]
    return title


def _key(model: type[LabModel[Any]], values: dict[str, Any]) -> str:
    return json.dumps({"model": model.info.name, **values}, sort_keys=True, default=str)
