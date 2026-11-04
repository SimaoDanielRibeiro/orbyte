"""O coordenador: distribui o orçamento pelos agentes e escolhe o vencedor, só com o treino.

A procura tem duas rondas:

1. **Triagem**: cada agente experimenta os seus modelos com os valores por omissão. É a
   ronda barata, que mostra que famílias se dão bem com estes dados. Os modelos vão à vez,
   uma família de cada vez: um orçamento curto chega a todas as famílias antes de chegar
   ao segundo modelo de qualquer uma. A referência (o modelo sugerido para a tarefa, como a
   regressão logística) vai primeiro, porque é o resultado que os outros têm de bater.
2. **Afinação**: os ``finalists`` melhores modelos da triagem passam. À vez, o agente de
   cada um propõe uma combinação nova de hiperparâmetros (procura aleatória), até o
   orçamento acabar. Se houver um agente de IA (ADR 0042), também tem a sua vez: olha para
   todos os resultados e propõe o próximo candidato, que se mede como os outros. Uma proposta
   que não serve é recusada sem gastar orçamento; à terceira seguida, ele sai da afinação.

Cada candidato é medido com **validação cruzada** nos folds do treino. O coordenador nunca
recebe o teste: recebe os folds, e é tudo o que vê. Escolher pelo teste faria do teste mais
um treino, e a nota final do vencedor deixava de ser honesta.

O vencedor é o candidato com a melhor média; num empate, o de menor desvio padrão (o mais
estável) e depois o que veio primeiro (a triagem vem antes da afinação).

A conversa conta a procura como uma troca, com quem fala a quem: o coordenador anuncia cada
ronda aos agentes, dá a vez a um agente de cada vez (um pedido, com o número do candidato que
vai sair dele), o agente responde-lhe com o resultado, e no fim o coordenador passa o melhor ao
vencedor. O histórico visual da interface desenha estas trocas pela ordem.

E conta-a por palavras (ADR 0049): o pedido diz porquê (a referência, a primeira impressão, a
distância ao primeiro), e a resposta diz o que se experimentou, o que o resultado quer dizer e
se ajudou. Os números ficam no cartão do candidato, ao lado.
"""

import math
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ml_lab.core.automl.ai_agent import MAX_REFUSALS, AiAgent
from ml_lab.core.automl.model_agent import ModelAgent, describe_changes
from ml_lab.core.automl.objective import Objective
from ml_lab.core.automl.search_log import (
    COORDINATOR,
    TO_AI,
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
from ml_lab.core.foundation.errors import LanguageModelError
from ml_lab.core.models.base import LabModel
from ml_lab.core.training.control import TrainingControl
from ml_lab.formatting import format_count, format_list, format_quantity
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
        ai: O agente de IA, se entrar na afinação.
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
        ai: AiAgent | None = None,
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
        self._ai = ai
        self._ai_refusals = 0
        self._candidates: list[Candidate] = []
        self._stopped = False

    def search(self) -> SearchOutcome:
        """Corre as duas rondas e escolhe o vencedor."""
        self._screening()
        if self._left() > 0 and not self._stopped:
            self._tuning()
        ranked = rank(self._candidates, self._objective)
        winner = ranked[0] if ranked else None
        # Parar é parar tudo (ADR 0051): não se anuncia um vencedor, porque ele não vai treinar
        # nem ao teste. O melhor até aí fica no quadro.
        if not self._stopped:
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
            f"Round 1, screening: {format_quantity(len(planned), 'model', 'models')} from "
            f"{format_quantity(families, 'family', 'families')}, each with the default values. "
            "It tells which families get on well with this data, before spending the budget "
            "on tuning.",
            recipients=to_agents(agent.family for agent, _ in planned),
        )
        if len(planned) < len(order):
            left_out = ", ".join(model.info.display_name for _, model in order[len(planned) :])
            self._say(
                f"The budget only covers {len(planned)} of the {len(order)} models: left out "
                f"are {left_out}."
            )
        for agent, model in planned:
            if not self._may_continue():
                return
            name = model.info.display_name
            if model.info.name == self._baseline:
                request = (
                    f"start with {name}, with its default settings: it's the reference, the "
                    "result the other models must beat."
                )
            else:
                request = (
                    f"measure {name} with its default settings: a first impression, before any "
                    "tuning."
                )
            self._ask(agent, request)
            candidate = self._evaluate(agent, model, agent.defaults(model), SearchPhase.SCREENING)
            self._agent_says(agent, candidate, None)

    def _tuning(self) -> None:
        finalists = self._choose_finalists()
        if not finalists:
            self._say("No model got through the screening: there's nothing to tune.")
            return
        names = format_list(
            [
                f"{model.info.display_name} at {self._objective.written(self._best(model))}"
                for _, model in finalists
            ]
        )
        self._say(
            f"Through to the tuning: {names}. In turn, each one's agent tries new values for its "
            "settings, drawn at random from the ranges worth searching (random search), until "
            f"the {format_count(self._left())} candidates left in the budget are spent.",
            recipients=to_agents(agent.family for agent, _ in finalists),
        )
        self._log.enter(SearchPhase.TUNING)
        # Na roda, ``None`` é a vez do agente de IA: entra depois dos agentes de família.
        rotation: deque[Entry | None] = deque(finalists)
        if self._ai is not None:
            rotation.append(None)
            self._say(
                f"{self._ai.name}, you join the tuning too. On your turn, look at all the results "
                "and propose the next candidate among the finalists: it's measured like the "
                "others, with cross-validation, and counts towards the budget.",
                recipients=TO_AI,
            )
        while rotation and self._left() > 0:
            entry = rotation.popleft()
            if entry is None:
                if self._ai_turn(finalists):
                    rotation.append(None)
                continue
            agent, model = entry
            turn = f"it's your turn to tune {model.info.display_name}."
            if agent.space(model).is_empty:
                self._ask(agent, turn, numbered=False)
                self._agent_says(
                    agent,
                    None,
                    None,
                    text=f"{model.info.display_name} has no settings to tune: it stays in the "
                    "race with its default values.",
                )
                continue
            values = agent.propose(model, self._rng)
            if values is None:
                self._ask(agent, turn, numbered=False)
                self._agent_says(
                    agent,
                    None,
                    None,
                    text=f"I've already tried every combination of {model.info.display_name}'s "
                    "settings: there's nothing new to measure.",
                )
                continue
            if not self._may_continue():
                return
            self._ask(agent, self._tuning_request(model))
            best_before = self._best(model)
            candidate = self._evaluate(agent, model, values, SearchPhase.TUNING)
            self._agent_says(agent, candidate, best_before)
            rotation.append((agent, model))

    def _ai_turn(self, finalists: Sequence[Entry]) -> bool:
        """A vez do agente de IA: pensa, propõe, e o coordenador mede (ou recusa).

        Returns:
            Se o agente de IA fica na afinação (sai se o modelo de linguagem falhar, ou à
            terceira proposta recusada seguida).
        """
        ai = self._ai
        assert ai is not None
        if not self._may_continue():
            return False
        number = self._log.next_number
        self._say(
            f"{ai.name}, it's your turn: choose the next candidate among the finalists.",
            kind=MessageKind.REQUEST,
            recipients=TO_AI,
            candidate=number,
        )
        # O raciocínio vai para o registo aos bocados: quem acompanha vê-o a crescer.
        self._log.start_thinking(ai.name, number)
        try:
            turn = ai.propose(finalists, self._candidates, on_thinking=self._log.think)
        except LanguageModelError as error:
            self._ai_says(f"I couldn't answer: {error} The tuning goes on without me.")
            return False
        finally:
            self._log.stop_thinking()
        thinking = turn.reply.thinking.strip()
        if thinking:
            self._log.say(
                AgentRole.AI, ai.name, thinking, candidate=number, kind=MessageKind.THOUGHT
            )
        proposal = turn.proposal
        if proposal is None:
            self._ai_refusals += 1
            out = self._ai_refusals >= MAX_REFUSALS
            self._say(
                f"{ai.name}, your proposal won't do: {turn.problem}. I won't measure it, and it "
                "doesn't count towards the budget."
                + (f" That's {MAX_REFUSALS} in a row: you leave the tuning." if out else ""),
                recipients=TO_AI,
            )
            return not out
        self._ai_refusals = 0
        model = proposal.model
        defaults = proposal.agent.defaults(model)
        changes = {k: v for k, v in proposal.values.items() if defaults.get(k) != v}
        settings = describe_changes(changes) or "the default values"
        self._ai_says(
            f"I propose {model.info.display_name} with {settings}. {proposal.reason}",
            candidate=number,
        )
        best_before = self._best(model)
        candidate = self._evaluate(
            proposal.agent,
            model,
            proposal.values,
            SearchPhase.TUNING,
            author=ai.name,
            reason=proposal.reason,
        )
        leader_before, first = self._before(candidate)
        self._ai_says(
            proposal.agent.report(
                candidate, best_before, self._objective, leader_before=leader_before, first=first
            ),
            candidate=candidate.number,
        )
        return True

    def _tuning_request(self, model: type[LabModel[Any]]) -> str:
        """O pedido da afinação: o modelo, onde está em relação ao primeiro, e o que se quer."""
        name, objective = model.info.display_name, self._objective
        best = self._best(model)
        ranked = rank(self._candidates, objective)
        leader = ranked[0] if ranked else None
        if best is None or leader is None or leader.mean is None:
            return (
                f"it's your turn to tune {name}: draw new values for its settings and see if it "
                "does better."
            )
        if leader.model == model.info.name:
            return (
                f"it's your turn to tune {name}. It leads the search with "
                f"{objective.written(best)}: draw new values for its settings and see if it goes "
                "further."
            )
        if objective.written(best) == objective.written(leader.mean):
            return (
                f"it's your turn to tune {name}. Its best so far, {objective.written(best)}, is "
                f"level with the leader, {leader.model_display_name}: draw new values for its "
                "settings and see if it pulls ahead."
            )
        return (
            f"it's your turn to tune {name}. Its best so far is {objective.written(best)}, "
            f"{objective.gap(leader.mean, best)} behind the leader, {leader.model_display_name} "
            f"({objective.written(leader.mean)}): draw new values for its settings and see if it "
            "catches up."
        )

    def _before(self, candidate: Candidate) -> tuple[float | None, bool]:
        """O melhor resultado da procura antes de ``candidate``, e se ele é o primeiro."""
        previous = [c for c in self._candidates if c.number != candidate.number]
        ranked = rank(previous, self._objective)
        return (ranked[0].mean if ranked else None), not previous

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
        *,
        author: str | None = None,
        reason: str | None = None,
    ) -> Candidate:
        """Mede um candidato com validação cruzada.

        Args:
            agent: O agente da família do modelo (toma nota da combinação).
            model: O modelo.
            values: Os hiperparâmetros completos.
            phase: A ronda.
            author: Quem o propôs, se não for o agente da família (o agente de IA).
            reason: Porque foi proposto (só o agente de IA o diz).
        """
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
            ai=author is not None,
            reason=reason,
        )
        self._log.start(
            WorkInProgress(
                candidate=candidate.number,
                phase=phase,
                family=info.family,
                author=author or agent.name,
                model=info.name,
                model_display_name=info.display_name,
                changes=candidate.changes,
                fold=0,
                folds=len(self._folds),
                ai=author is not None,
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
            self._say("No candidate made it to the end of the cross-validation: there's no winner.")
            return
        objective = self._objective
        agent = next((a for a in self._agents if winner.family is a.family), None)
        model = next(
            (m for m in (agent.models if agent else ()) if m.info.name == winner.model), None
        )
        if not winner.changes:
            settings = "its default settings"
        elif agent is not None and model is not None:
            settings = agent.explain_changes(model, winner.changes)
        else:
            settings = describe_changes(winner.changes)
        text = (
            f"The best in the cross-validation is {winner.model_display_name}, with {settings}: "
            f"{objective.written(winner.mean)} ± {objective.written(winner.std)} "
            f"({objective.label}, {objective.direction})."
        )
        reading = objective.reading(winner.mean)
        if reading:
            text += f" {reading}."
        second = ranked[1] if len(ranked) > 1 else None
        if second is not None and second.mean is not None and winner.mean is not None:
            if objective.written(second.mean) == objective.written(winner.mean):
                # Um empate à vista: o vencedor ganha pelo desvio padrão ou por ter vindo antes.
                text += (
                    f" The second, {second.model_display_name}, is level with it "
                    f"({objective.written(second.mean)}): with a difference this small, either is "
                    "as good a choice."
                )
            else:
                gap = objective.gap(winner.mean, second.mean)
                text += (
                    f" The second is {second.model_display_name} "
                    f"({objective.written(second.mean)}), {gap} behind."
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

    def _ai_says(self, text: str, *, candidate: int | None = None) -> None:
        """O agente de IA responde ao coordenador."""
        assert self._ai is not None
        self._log.say(
            AgentRole.AI,
            self._ai.name,
            text,
            candidate=candidate,
            kind=MessageKind.REPLY,
            recipients=TO_COORDINATOR,
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
            leader_before, first = self._before(candidate)
            text = agent.report(
                candidate, best_before, self._objective, leader_before=leader_before, first=first
            )
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
            measured = format_quantity(len(self._candidates), "candidate", "candidates")
            self._say(
                f"You asked to stop, after {measured}: nothing more is measured, and no winner "
                "is trained."
            )
        return not self._stopped
