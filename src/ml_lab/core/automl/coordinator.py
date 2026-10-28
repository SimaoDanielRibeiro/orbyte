"""O coordenador: distribui o orçamento pelos agentes e escolhe o vencedor, só com o treino.

A procura tem duas rondas:

1. **Triagem**: cada agente experimenta os seus modelos com os valores por omissão. É a
   ronda barata, que mostra que famílias se dão bem com estes dados. Os modelos vão à vez,
   uma família de cada vez: um orçamento curto chega a todas as famílias antes de chegar
   ao segundo modelo de qualquer uma. A referência (o modelo sugerido para a tarefa, como a
   regressão logística) vai primeiro, porque é o resultado que os outros têm de bater.
2. **Afinação**: os ``finalists`` melhores modelos da triagem passam. À vez, o agente de
   cada um propõe uma combinação nova de hiperparâmetros (procura aleatória), até o
   orçamento acabar.

Cada candidato é medido com **validação cruzada** nos folds do treino. O coordenador nunca
recebe o teste: recebe os folds, e é tudo o que vê. Escolher pelo teste faria do teste mais
um treino, e a nota final do vencedor deixava de ser honesta.

O vencedor é o candidato com a melhor média; num empate, o de menor desvio padrão (o mais
estável) e depois o que veio primeiro (a triagem vem antes da afinação).

A conversa conta a procura como uma troca, com quem fala a quem: o coordenador anuncia cada
ronda aos agentes, dá a vez a um agente de cada vez (um pedido, com o número do candidato que
vai sair dele), o agente responde-lhe com o resultado, e no fim o coordenador passa o melhor ao
vencedor. O histórico visual da interface desenha estas trocas pela ordem.
"""

import math
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ml_lab.core.automl.model_agent import ModelAgent, describe_changes
from ml_lab.core.automl.objective import Objective
from ml_lab.core.automl.search_log import (
    COORDINATOR,
    TO_COORDINATOR,
    TO_WINNER,
    AgentRole,
    Candidate,
    MessageKind,
    Participant,
    SearchLog,
    SearchPhase,
    WorkInProgress,
    to_agents,
)
from ml_lab.core.data.dataset import Dataset
from ml_lab.core.evaluation.validation import cross_validate
from ml_lab.core.models.base import LabModel
from ml_lab.core.training.control import TrainingControl
from ml_lab.formatting import format_count, format_quantity
from ml_lab.logging import get_logger

logger = get_logger(__name__)

type Entry = tuple[ModelAgent, type[LabModel[Any]]]
"""Um modelo e o agente que responde por ele."""


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """O que a procura encontrou.

    Attributes:
        candidates: Todos os candidatos, pela ordem em que foram medidos.
        ranking: Os números dos candidatos medidos, do melhor para o pior.
        winner: O melhor candidato, se algum foi medido.
        stopped: Se a procura parou a pedido antes de gastar o orçamento.
    """

    candidates: tuple[Candidate, ...]
    ranking: tuple[int, ...]
    winner: Candidate | None
    stopped: bool = False


def rank(candidates: Sequence[Candidate], objective: Objective) -> tuple[Candidate, ...]:
    """Os candidatos medidos, do melhor para o pior (os que falharam ficam de fora)."""
    measured = [candidate for candidate in candidates if candidate.mean is not None]
    return tuple(
        sorted(
            measured,
            key=lambda c: (objective.key(c.mean), math.inf if c.std is None else c.std, c.number),
        )
    )


class Coordinator:
    """Gere a procura: as rondas, o orçamento e a escolha do vencedor.

    Args:
        objective: A métrica a otimizar.
        agents: Os agentes de modelo.
        folds: Os folds da validação cruzada, só do treino (de ``prepare_folds``).
        budget: Quantos candidatos medir, no máximo (cada um treina uma vez por fold).
        finalists: Quantos modelos passam da triagem à afinação.
        model_seed: A seed dos modelos (a mesma do run final, para o vencedor treinar igual).
        search_seed: A seed da procura aleatória.
        log: Onde escrever a conversa e os candidatos.
        control: Os pedidos de pausa e de paragem, se a procura correr em segundo plano.
        baseline: O modelo de referência, que vai primeiro na triagem.
    """

    def __init__(
        self,
        objective: Objective,
        agents: Sequence[ModelAgent],
        folds: Sequence[Dataset],
        *,
        budget: int,
        finalists: int,
        model_seed: int,
        search_seed: int,
        log: SearchLog,
        control: TrainingControl | None = None,
        baseline: str | None = None,
    ) -> None:
        self._objective = objective
        self._agents = tuple(agents)
        self._folds = tuple(folds)
        self._budget = budget
        self._finalists = finalists
        self._model_seed = model_seed
        self._rng = np.random.default_rng(search_seed)
        self._log = log
        self._control = control
        self._baseline = baseline
        self._candidates: list[Candidate] = []
        self._stopped = False

    def search(self) -> SearchOutcome:
        """Corre as duas rondas e escolhe o vencedor."""
        self._screening()
        if self._left() > 0 and not self._stopped:
            self._tuning()
        ranked = rank(self._candidates, self._objective)
        winner = ranked[0] if ranked else None
        self._announce(winner, ranked)
        return SearchOutcome(
            candidates=tuple(self._candidates),
            ranking=tuple(candidate.number for candidate in ranked),
            winner=winner,
            stopped=self._stopped,
        )

    # --- as rondas ---------------------------------------------------------------------

    def _screening(self) -> None:
        self._log.enter(SearchPhase.SCREENING)
        order = self._screening_order()
        planned = order[: self._budget]
        families = len({agent.family for agent, _ in planned})
        self._say(
            f"Ronda 1, triagem: {format_quantity(len(planned), 'modelo', 'modelos')} de "
            f"{format_quantity(families, 'família', 'famílias')}, cada um com os valores por "
            "omissão. Diz que famílias se dão bem com estes dados, antes de gastar o "
            "orçamento a afinar.",
            recipients=to_agents(agent.family for agent, _ in planned),
        )
        if len(planned) < len(order):
            left_out = ", ".join(model.info.display_name for _, model in order[len(planned) :])
            self._say(
                f"O orçamento só chega para {len(planned)} dos {len(order)} modelos: ficam de "
                f"fora {left_out}."
            )
        for agent, model in planned:
            if not self._may_continue():
                return
            self._ask(agent, f"mede {model.info.display_name} com os valores por omissão.")
            candidate = self._evaluate(agent, model, agent.defaults(model), SearchPhase.SCREENING)
            self._agent_says(agent, candidate, None)

    def _tuning(self) -> None:
        finalists = self._choose_finalists()
        if not finalists:
            self._say("Nenhum modelo passou a triagem: não há nada para afinar.")
            return
        names = ", ".join(
            f"{model.info.display_name} ({self._objective.written(self._best(model))})"
            for _, model in finalists
        )
        self._say(
            f"Passam à afinação: {names}. À vez, o agente de cada um propõe outra combinação "
            "de hiperparâmetros, sorteada do espaço de procura do modelo (procura aleatória), "
            f"até gastar os {format_count(self._left())} candidatos que sobram do orçamento.",
            recipients=to_agents(agent.family for agent, _ in finalists),
        )
        self._log.enter(SearchPhase.TUNING)
        rotation: deque[Entry] = deque(finalists)
        while rotation and self._left() > 0:
            agent, model = rotation.popleft()
            turn = f"é a tua vez de afinar {model.info.display_name}."
            if agent.space(model).is_empty:
                self._ask(agent, turn, numbered=False)
                self._agent_says(
                    agent,
                    None,
                    None,
                    text=f"{model.info.display_name} não tem hiperparâmetros para afinar: fica "
                    "na corrida com os valores por omissão.",
                )
                continue
            values = agent.propose(model, self._rng)
            if values is None:
                self._ask(agent, turn, numbered=False)
                self._agent_says(
                    agent,
                    None,
                    None,
                    text=f"Já experimentei todas as combinações de {model.info.display_name}.",
                )
                continue
            if not self._may_continue():
                return
            self._ask(agent, turn)
            best_before = self._best(model)
            candidate = self._evaluate(agent, model, values, SearchPhase.TUNING)
            self._agent_says(agent, candidate, best_before)
            rotation.append((agent, model))

    def _screening_order(self) -> list[Entry]:
        """Os modelos da triagem: uma família de cada vez, com a referência à frente."""
        queues = [deque((agent, model) for model in agent.models) for agent in self._agents]
        order: list[Entry] = []
        while any(queues):
            for queue in queues:
                if queue:
                    order.append(queue.popleft())
        baseline = [entry for entry in order if entry[1].info.name == self._baseline]
        return baseline + [entry for entry in order if entry not in baseline]

    def _choose_finalists(self) -> list[Entry]:
        """Os ``finalists`` melhores modelos da triagem (cada modelo conta uma vez)."""
        entries = {
            model.info.name: (agent, model) for agent in self._agents for model in agent.models
        }
        chosen: list[Entry] = []
        for candidate in rank(self._candidates, self._objective):
            entry = entries[candidate.model]
            if entry not in chosen:
                chosen.append(entry)
            if len(chosen) == self._finalists:
                break
        return chosen

    # --- medir um candidato ------------------------------------------------------------

    def _evaluate(
        self,
        agent: ModelAgent,
        model: type[LabModel[Any]],
        values: dict[str, Any],
        phase: SearchPhase,
    ) -> Candidate:
        info = model.info
        agent.remember(model, values)
        defaults = agent.defaults(model)
        candidate = Candidate(
            number=self._log.next_number,
            phase=phase,
            family=info.family,
            model=info.name,
            model_display_name=info.display_name,
            hyperparameters=values,
            changes={name: value for name, value in values.items() if defaults.get(name) != value},
        )
        self._log.start(
            WorkInProgress(
                candidate=candidate.number,
                phase=phase,
                family=info.family,
                author=agent.name,
                model=info.name,
                model_display_name=info.display_name,
                changes=candidate.changes,
                fold=0,
                folds=len(self._folds),
            )
        )
        start = time.perf_counter()
        try:
            hyperparameters = info.hyperparameters.parse(values, model=info.name)
            cv = cross_validate(
                model,
                hyperparameters,
                self._model_seed,
                self._folds,
                self._objective.metric,
                on_fold=self._log.progress,
            )
        except Exception as error:  # um candidato que falha não para a procura: fica com a razão
            logger.warning("o candidato falhou", extra={"model": info.name, "error": str(error)})
            candidate = replace(
                candidate, seconds=time.perf_counter() - start, error=str(error) or repr(error)
            )
        else:
            trains = [fold.train for fold in cv.folds if fold.train is not None]
            candidate = replace(
                candidate,
                mean=cv.mean,
                std=cv.std,
                train_mean=float(np.mean(trains)) if trains else None,
                folds=tuple(fold.test for fold in cv.folds),
                seconds=time.perf_counter() - start,
            )
        self._candidates.append(candidate)
        self._log.add(candidate)
        return candidate

    # --- a conversa --------------------------------------------------------------------

    def _announce(self, winner: Candidate | None, ranked: Sequence[Candidate]) -> None:
        """Escolhe o vencedor e passa-o à etapa seguinte (o treino com tudo e o teste)."""
        if winner is None:
            self._say("Nenhum candidato chegou ao fim da validação cruzada: não há vencedor.")
            return
        objective = self._objective
        settings = describe_changes(winner.changes) or "os valores por omissão"
        text = (
            f"O melhor na validação cruzada é {winner.model_display_name}, com {settings}: "
            f"{objective.written(winner.mean)} ± {objective.written(winner.std)} "
            f"({objective.label}, {objective.direction})."
        )
        second = ranked[1] if len(ranked) > 1 else None
        if second is not None and second.mean is not None and winner.mean is not None:
            gap = objective.gap(winner.mean, second.mean)
            text += (
                f" O segundo é {second.model_display_name} ({objective.written(second.mean)}), "
                f"a {gap}."
            )
        self._say(text, kind=MessageKind.DECISION, recipients=TO_WINNER, candidate=winner.number)

    def _say(
        self,
        text: str,
        *,
        kind: MessageKind = MessageKind.ANNOUNCEMENT,
        recipients: tuple[Participant, ...] | None = None,
        candidate: int | None = None,
    ) -> None:
        """O coordenador fala; sem destinatários, fala para todos os agentes."""
        self._log.say(
            AgentRole.COORDINATOR,
            COORDINATOR,
            text,
            candidate=candidate,
            kind=kind,
            recipients=(
                to_agents(agent.family for agent in self._agents)
                if recipients is None
                else recipients
            ),
        )

    def _ask(self, agent: ModelAgent, request: str, *, numbered: bool = True) -> None:
        """O coordenador dá a vez a um agente (com o número do candidato que vai sair dela)."""
        self._say(
            f"{agent.name}, {request}",
            kind=MessageKind.REQUEST,
            recipients=to_agents([agent.family]),
            candidate=self._log.next_number if numbered else None,
        )

    def _agent_says(
        self,
        agent: ModelAgent,
        candidate: Candidate | None,
        best_before: float | None,
        *,
        text: str | None = None,
    ) -> None:
        if text is None:
            assert candidate is not None
            text = agent.report(candidate, best_before, self._objective)
        self._log.say(
            AgentRole.MODEL,
            agent.name,
            text,
            family=agent.family,
            candidate=None if candidate is None else candidate.number,
            kind=MessageKind.REPLY,
            recipients=TO_COORDINATOR,
        )

    # --- o orçamento e o controlo --------------------------------------------------------

    def _left(self) -> int:
        return self._budget - len(self._candidates)

    def _best(self, model: type[LabModel[Any]]) -> float | None:
        """O melhor resultado de ``model`` até agora."""
        scores = [c.mean for c in self._candidates if c.model == model.info.name]
        return min(scores, key=self._objective.key, default=None)

    def _may_continue(self) -> bool:
        """Espera durante uma pausa; ``False`` se foi pedido para parar."""
        if self._control is None:
            return True
        self._control.wait_while_paused()
        if self._control.stop_requested and not self._stopped:
            self._stopped = True
            measured = format_quantity(len(self._candidates), "candidato", "candidatos")
            self._say(f"Pediste para parar, ao fim de {measured}: o vencedor é o melhor até aqui.")
        return not self._stopped
