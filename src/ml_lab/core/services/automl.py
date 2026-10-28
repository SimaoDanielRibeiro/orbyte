"""Serviço dos agentes (fase 9): o modo automático, que procura o melhor modelo para um pedido.

É o ecrã Agentes, e segue o fluxo dos outros serviços (ADR 0025):

1. **o plano** (falha logo, antes de começar): a seed raiz e as seeds de cada parte (ADR
   0006); o agente de dados decide a preparação, o objetivo valida-se, os agentes de modelo
   formam-se e os dados preparam-se uma vez, para um problema nos dados aparecer já;
2. **a procura**, em segundo plano: os folds do treino (uma amostra, se houver linhas a
   mais) e as duas rondas do coordenador. Cada candidato medido é um passo do trabalho, por
   isso a procura pausa e pára como um treino;
3. **o vencedor** treina como um run normal do ``ExperimentService``: usa o teste pela
   primeira e única vez e fica no histórico, ao lado dos runs feitos à mão;
4. **o resultado**: o quadro dos candidatos, a conversa dos agentes e a leitura do vencedor
   contra a referência e contra o teste.

``run`` faz tudo e devolve o resultado; ``start`` corre em segundo plano, e ``job``,
``updates``, ``pause``, ``resume``, ``cancel`` e ``result`` servem quem só sabe o
identificador da procura (a API, ADR 0022). ``train_candidate`` é a ponte para o modo
manual: treina um candidato do quadro como um run normal, para o ver no Treinar e avaliar.
"""

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from ml_lab.config import Settings, get_settings
from ml_lab.core.automl.coordinator import Coordinator, SearchOutcome, rank
from ml_lab.core.automl.data_agent import DataAgent, DataDecision
from ml_lab.core.automl.model_agent import (
    ModelAgent,
    SkippedModel,
    describe_changes,
    describe_skipped,
    model_agents,
)
from ml_lab.core.automl.objective import (
    OBJECTIVES,
    Objective,
    objective_for,
    supports,
    unsupported_reason,
)
from ml_lab.core.automl.search_log import (
    COORDINATOR,
    DATA_AGENT,
    AgentMessage,
    AgentRole,
    Candidate,
    SearchLog,
    SearchPhase,
    WorkInProgress,
)
from ml_lab.core.automl.search_space import search_space
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.evaluation.explanations import METRIC_EXPLANATIONS
from ml_lab.core.evaluation.metrics.base import METRICS
from ml_lab.core.evaluation.report import MetricResult
from ml_lab.core.foundation.errors import (
    CandidateNotFoundError,
    ColumnNotFoundError,
    InvalidParameterError,
    MLLabError,
    SearchNotFoundError,
)
from ml_lab.core.foundation.events import StepCompleted
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import ModelFamily
from ml_lab.core.schemas.automl import AutoMLRequest, SearchSettings
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.services.catalog import SUGGESTED_MODELS
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import MODEL_SEED, ExperimentResult, ExperimentService
from ml_lab.core.services.history import HistoryService
from ml_lab.core.training.control import TrainingControl
from ml_lab.core.training.jobs import JobRegistry, JobState, TrainingJob
from ml_lab.core.training.recorder import SnapshotRecorder, StepRecord
from ml_lab.formatting import format_count, format_quantity
from ml_lab.logging import get_logger

logger = get_logger(__name__)

SEARCH_SEED: Final = "automl"
"""Nome do componente da procura aleatória: a sua seed é ``seeds.seed_for("automl")``."""

KEPT_SEARCHES: Final = 5
"""Quantas procuras ficam em memória (com a conversa e o quadro dos candidatos)."""

DOCS_FOLDER: Final = Path("docs") / "algorithms"
DOC: Final = "automl.md"
"""A ficha didática dos agentes, em ``docs/algorithms/``."""


@dataclass(frozen=True, slots=True)
class ObjectiveOption:
    """Uma métrica que os agentes sabem otimizar.

    Attributes:
        name: O nome no registo.
        label: O nome a mostrar.
        higher_is_better: Se maior é melhor.
        as_percent: Se se mostra como percentagem.
        needs_proba: Se só entram os modelos que dão probabilidades.
        what: O que mede.
    """

    name: str
    label: str
    higher_is_better: bool
    as_percent: bool
    needs_proba: bool
    what: str


@dataclass(frozen=True, slots=True)
class AgentModel:
    """Um modelo de um agente e o que o agente procura nele.

    Attributes:
        name: O modelo, no registo.
        display_name: O nome a mostrar.
        searches: O que se procura em cada hiperparâmetro ("C: de 0,1 a 100…"); vazio se o
            modelo só se experimenta com os valores por omissão.
    """

    name: str
    display_name: str
    searches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelAgentCard:
    """Um agente de modelo, pronto a mostrar.

    Attributes:
        family: A família de que trata.
        name: O nome a mostrar ("Agente · Árvores de decisão").
        models: Os modelos que experimenta.
    """

    family: ModelFamily
    name: str
    models: tuple[AgentModel, ...]


@dataclass(frozen=True, slots=True)
class AutoMLOptions:
    """O que se pode pedir aos agentes numa tarefa (o ecrã Agentes monta o pedido com isto).

    Attributes:
        task_type: A tarefa.
        supported: Se os agentes trabalham nesta tarefa.
        reason: Se não trabalham, porquê.
        objectives: As métricas que sabem otimizar; a primeira é a principal.
        agents: Os agentes de modelo, com os modelos de cada um.
        skipped: Os modelos da tarefa que ficam de fora, com a razão.
        settings_schema: O JSON Schema do tamanho da procura (de onde se gera o formulário).
        settings: Os valores por omissão do tamanho da procura.
        doc: A ficha didática dos agentes, em Markdown.
    """

    task_type: TaskType
    supported: bool
    reason: str | None
    objectives: tuple[ObjectiveOption, ...]
    agents: tuple[ModelAgentCard, ...]
    skipped: tuple[SkippedModel, ...]
    settings_schema: dict[str, Any]
    settings: SearchSettings
    doc: str


@dataclass(frozen=True, slots=True)
class AutoMLPlan:
    """O que os agentes vão fazer com um pedido (o ecrã mostra-o antes de começar).

    Attributes:
        dataset_id: O dataset.
        dataset_name: O nome a mostrar do dataset.
        target: A coluna a prever.
        task_type: A tarefa.
        preparation: A preparação decidida (ou adotada) pelo agente de dados.
        decisions: As decisões do agente de dados, com a razão de cada uma.
        adopted: Se o agente de dados adotou uma preparação feita à mão.
        objective: A métrica a otimizar.
        agents: Os agentes de modelo chamados.
        skipped: Os modelos que ficam de fora, com a razão.
        baseline: O modelo de referência (o que os outros têm de bater), se entrar.
        train_rows: As linhas de treino.
        test_rows: As linhas de teste (guardadas até ao fim).
        search_rows: As linhas que a validação cruzada usa (uma amostra, se houver a mais).
        settings: O tamanho da procura.
    """

    dataset_id: str
    dataset_name: str
    target: str
    task_type: TaskType
    preparation: PreparationConfig
    decisions: tuple[DataDecision, ...]
    adopted: bool
    objective: ObjectiveOption
    agents: tuple[ModelAgentCard, ...]
    skipped: tuple[SkippedModel, ...]
    baseline: str | None
    train_rows: int
    test_rows: int
    search_rows: int
    settings: SearchSettings

    @property
    def sampled(self) -> bool:
        """Se a validação cruzada usa uma amostra do treino."""
        return self.search_rows < self.train_rows


@dataclass(frozen=True, slots=True)
class AutoMLUpdate:
    """O que mudou numa procura desde a última pergunta (o que a API transmite).

    Attributes:
        state: O retrato do trabalho agora (o progresso conta candidatos).
        phase: A fase da procura.
        objective: A métrica que os agentes otimizam (para mostrar os valores).
        candidates: Os candidatos novos (depois de ``after_step``), pela ordem.
        messages: As mensagens novas da conversa (depois de ``after_message``).
        leader: O melhor candidato até agora.
        working: O candidato que um agente está a medir agora, e em que fold vai
            (``None`` entre candidatos e fora das rondas).
    """

    state: JobState
    phase: SearchPhase
    objective: ObjectiveOption
    candidates: tuple[Candidate, ...] = ()
    messages: tuple[AgentMessage, ...] = ()
    leader: Candidate | None = None
    working: WorkInProgress | None = None

    @property
    def steps(self) -> tuple[StepRecord, ...]:
        """Os candidatos novos como passos (o WebSocket usa-os para saber onde vai)."""
        return tuple(StepRecord(c.number, None, {}) for c in self.candidates)

    @property
    def snapshot_step(self) -> int | None:
        """O número da última mensagem nova: o segundo cursor do WebSocket."""
        return self.messages[-1].number if self.messages else None


@dataclass(frozen=True, slots=True)
class AutoMLResult:
    """O resultado de uma procura, pronto a mostrar.

    Attributes:
        search_id: O identificador da procura.
        request: O pedido, com a seed já resolvida: repete a procura tal e qual.
        plan: O que os agentes decidiram antes de procurar.
        candidates: Todos os candidatos, pela ordem em que foram medidos.
        ranking: Os números dos candidatos medidos, do melhor para o pior.
        winner: O número do vencedor, se houver.
        baseline: O número do candidato de referência (com os valores por omissão), se houver.
        messages: A conversa dos agentes, do princípio ao fim.
        run_id: O run do vencedor, treinado com o treino todo (está no histórico).
        test: O objetivo no teste, para o vencedor (a única vez que o teste se usa).
        notices: A leitura do resultado.
        duration_seconds: Quanto tempo demorou, com o treino do vencedor.
        stopped_early: Se a procura parou a pedido antes de gastar o orçamento.
    """

    search_id: str
    request: AutoMLRequest
    plan: AutoMLPlan
    candidates: tuple[Candidate, ...]
    ranking: tuple[int, ...]
    winner: int | None
    baseline: int | None
    messages: tuple[AgentMessage, ...]
    run_id: str | None
    test: MetricResult | None
    notices: tuple[Notice, ...]
    duration_seconds: float
    stopped_early: bool = False

    @property
    def seed(self) -> int:
        """A seed raiz usada."""
        assert self.request.seed is not None  # o serviço resolve-a sempre antes de procurar
        return self.request.seed

    def candidate(self, number: int) -> Candidate:
        """O candidato com o número ``number``.

        Raises:
            CandidateNotFoundError: Se a procura não tiver esse candidato.
        """
        if not 1 <= number <= len(self.candidates):
            raise CandidateNotFoundError(self.search_id, number, len(self.candidates))
        return self.candidates[number - 1]

    @property
    def winner_candidate(self) -> Candidate | None:
        """O vencedor, se houver."""
        return None if self.winner is None else self.candidate(self.winner)


@dataclass(frozen=True, slots=True, eq=False)
class _Plan:
    """O que fica decidido antes de procurar."""

    search_id: str
    request: AutoMLRequest
    seeds: SeedManager
    preview: AutoMLPlan
    objective: Objective
    agents: tuple[ModelAgent, ...]
    log: SearchLog
    recorder: SnapshotRecorder


class AutoMLService:
    """Os agentes: procuram o melhor modelo para um pedido, e explicam a escolha.

    Args:
        datasets: O serviço de datasets (o perfil, a preparação e os folds).
        experiments: O serviço de experiências (treina o vencedor e os candidatos à mão).
        history: O histórico, para dar ao run do vencedor um nome que diga de onde veio.
        settings: As definições (a raiz do projeto, onde está a ficha didática).
    """

    def __init__(
        self,
        datasets: DatasetService,
        experiments: ExperimentService,
        *,
        history: HistoryService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._datasets = datasets
        self._experiments = experiments
        self._history = history
        self._doc = (settings or get_settings()).home / DOCS_FOLDER / DOC
        self._jobs: JobRegistry[AutoMLResult] = JobRegistry(KEPT_SEARCHES)
        self._searches: OrderedDict[str, _Plan] = OrderedDict()
        self._results: OrderedDict[str, AutoMLResult] = OrderedDict()
        self._lock = threading.Lock()

    def options(self, task_type: TaskType, objective: str | None = None) -> AutoMLOptions:
        """O que se pode pedir aos agentes em ``task_type``: as métricas, os agentes e o orçamento.

        Args:
            task_type: A tarefa.
            objective: A métrica a otimizar (muda os modelos que entram); por omissão, a
                principal.

        Raises:
            InvalidParameterError: Se ``objective`` não for uma métrica que os agentes
                otimizem nesta tarefa.
        """
        options = AutoMLOptions(
            task_type=task_type,
            supported=supports(task_type),
            reason=None if supports(task_type) else unsupported_reason(task_type),
            objectives=(),
            agents=(),
            skipped=(),
            settings_schema=SearchSettings.model_json_schema(),
            settings=SearchSettings(),
            doc=self._doc.read_text(encoding="utf-8") if self._doc.is_file() else "",
        )
        if not options.supported:
            return options
        agents, skipped = model_agents(task_type, objective_for(task_type, objective))
        return replace(
            options,
            objectives=tuple(_objective_option(name) for name in OBJECTIVES[task_type]),
            agents=tuple(_agent_card(agent) for agent in agents),
            skipped=skipped,
        )

    def plan(self, request: AutoMLRequest) -> AutoMLPlan:
        """O que os agentes vão fazer com o pedido, sem começar: as decisões do agente de dados.

        Raises:
            ColumnNotFoundError: Se o dataset não tiver a coluna-alvo.
            DataError: Se o alvo não servir para prever, ou os dados não servirem.
            InvalidParameterError: Se a tarefa, a métrica ou as famílias não servirem.
        """
        return self._decide(request)[2]

    def run(self, request: AutoMLRequest) -> AutoMLResult:
        """Faz a procura toda e espera pelo resultado.

        Raises:
            ColumnNotFoundError, DataError, InvalidParameterError: Os problemas do pedido,
                antes de começar.
        """
        return self._search(self._plan(request), None)

    def start(self, request: AutoMLRequest) -> TrainingJob[AutoMLResult]:
        """Valida o pedido e procura em segundo plano (cada candidato medido é um passo).

        Raises:
            ColumnNotFoundError, DataError, InvalidParameterError: Os problemas do pedido,
                logo aqui.
        """
        plan = self._plan(request)
        job: TrainingJob[AutoMLResult] = TrainingJob(
            plan.search_id,
            lambda control: self._search(plan, control),
            plan.recorder,
            planned_steps=plan.request.settings.budget,
        )
        self._jobs.add(job.start())
        return job

    def job(self, search_id: str) -> JobState:
        """O estado de uma procura lançada com ``start``.

        Raises:
            SearchNotFoundError: Se não houver nenhuma procura com esse identificador.
        """
        return self._job(search_id).state()

    def pause(self, search_id: str) -> JobState:
        """Pede uma pausa no fim do candidato atual.

        Raises:
            SearchNotFoundError: Se não houver nenhuma procura com esse identificador.
        """
        job = self._job(search_id)
        job.pause()
        return job.state()

    def resume(self, search_id: str) -> JobState:
        """Retoma uma procura em pausa.

        Raises:
            SearchNotFoundError: Se não houver nenhuma procura com esse identificador.
        """
        job = self._job(search_id)
        job.resume()
        return job.state()

    def cancel(self, search_id: str) -> JobState:
        """Pede à procura que pare no fim do candidato atual (o vencedor é o melhor até aí).

        Raises:
            SearchNotFoundError: Se não houver nenhuma procura com esse identificador.
        """
        job = self._job(search_id)
        job.cancel()
        return job.state()

    def updates(self, search_id: str, after_step: int = 0, after_message: int = 0) -> AutoMLUpdate:
        """O que mudou numa procura desde a última pergunta.

        Args:
            search_id: A procura.
            after_step: O número do último candidato que quem pergunta já tem.
            after_message: O número da última mensagem que quem pergunta já tem.

        Raises:
            SearchNotFoundError: Se não houver nenhuma procura com esse identificador.
        """
        job = self._job(search_id)
        state = job.state()  # primeiro: se diz que acabou, o que se lê a seguir já está todo
        with self._lock:
            plan = self._searches.get(search_id)
        if plan is None:
            raise SearchNotFoundError(search_id)
        log = plan.log
        ranked = rank(log.candidates(), plan.objective)
        return AutoMLUpdate(
            state=state,
            phase=log.phase,
            objective=plan.preview.objective,
            candidates=log.candidates(after_step),
            messages=log.messages(after_message),
            leader=ranked[0] if ranked else None,
            working=log.working,
        )

    def search_plan(self, search_id: str) -> AutoMLPlan:
        """O plano de uma procura desta sessão: o que o agente de dados decidiu e os agentes.

        As atualizações só trazem o que muda; é com isto que quem acompanha a procura desenha
        os agentes (o fluxograma) e diz o que se está a prever, mesmo antes de ela acabar.

        Raises:
            SearchNotFoundError: Se a procura já não estiver em memória.
        """
        with self._lock:
            plan = self._searches.get(search_id)
            result = self._results.get(search_id)
        if plan is not None:
            return plan.preview
        if result is not None:
            return result.plan
        raise SearchNotFoundError(search_id)

    def result(self, search_id: str) -> AutoMLResult:
        """O resultado de uma procura desta sessão.

        Raises:
            TrainingInProgressError: Se a procura ainda decorrer.
            TrainingFailedError: Se a procura falhou.
            SearchNotFoundError: Se a procura já não estiver em memória.
        """
        if self._jobs.find(search_id) is not None:
            return self._jobs.result(search_id)
        with self._lock:
            result = self._results.get(search_id)
        if result is None:
            raise SearchNotFoundError(search_id)
        return result

    def train_candidate(self, search_id: str, number: int) -> TrainingJob[ExperimentResult]:
        """Treina um candidato do quadro como um run normal (o modo manual, a partir dos agentes).

        O run usa a preparação, os hiperparâmetros e a seed do candidato: segue-se no ecrã
        Treinar, avalia-se no teste e fica no histórico, como qualquer outro.

        Raises:
            SearchNotFoundError: Se a procura não estiver em memória.
            TrainingInProgressError: Se a procura ainda decorrer.
            CandidateNotFoundError: Se a procura não tiver esse candidato.
        """
        result = self.result(search_id)
        return self._experiments.start(_experiment(result, result.candidate(number)))

    # --- o plano -----------------------------------------------------------------------

    def _plan(self, request: AutoMLRequest) -> _Plan:
        """O plano de uma procura que vai correr (fica registado, para se acompanhar)."""
        request, seeds, preview, objective, agents = self._decide(request)
        search_id = f"search-{uuid.uuid4().hex[:8]}"
        recorder = SnapshotRecorder(search_id)

        def counted(candidate: Candidate) -> None:
            # Cada candidato medido é um passo do trabalho: é o que o progresso conta.
            recorder.receive(StepCompleted(run_id=search_id, step=candidate.number))

        plan = _Plan(
            search_id=search_id,
            request=request,
            seeds=seeds,
            preview=preview,
            objective=objective,
            agents=agents,
            log=SearchLog(counted),
            recorder=recorder,
        )
        with self._lock:
            self._searches[search_id] = plan
            self._forget_old_searches(keep=search_id)
        return plan

    def _forget_old_searches(self, *, keep: str) -> None:
        """Esquece os planos mais antigos, mas nunca o de uma procura que ainda decorre.

        O plano tem a conversa e os candidatos: sem ele, quem acompanha uma procura em curso
        deixava de a ver. Por isso só saem os planos de procuras que já acabaram.
        """
        excess = len(self._searches) - 2 * KEPT_SEARCHES
        finished = [
            search_id
            for search_id in self._searches
            if search_id != keep and ((job := self._jobs.find(search_id)) is None or job.done)
        ]
        for search_id in finished[: max(0, excess)]:
            del self._searches[search_id]

    def _decide(
        self, request: AutoMLRequest
    ) -> tuple[AutoMLRequest, SeedManager, AutoMLPlan, Objective, tuple[ModelAgent, ...]]:
        """O que fica decidido antes de procurar: a seed, os dados, o objetivo e os agentes."""
        seeds = SeedManager.create(request.seed)
        request = request.model_copy(update={"seed": seeds.seed})
        summary = self._datasets.load(request.dataset)
        columns = [column.name for column in summary.columns]
        if request.target not in columns:
            raise ColumnNotFoundError(request.target, columns)
        data = DataAgent().plan(
            self._datasets.profile(summary.id, request.target),
            self._datasets.suggest_task(summary.id, request.target),
            target=request.target,
            task_type=request.task_type,
            preparation=request.preparation,
            objective=request.objective,
        )
        objective = objective_for(data.task_type, data.objective)
        agents, skipped = model_agents(
            data.task_type,
            objective,
            families=request.families,
            include_didactic=request.include_didactic,
        )
        dataset = self._datasets.prepared(request.dataset, data.preparation, seeds.seed).dataset
        settings = request.settings
        train_rows = len(dataset.train)
        search_rows = min(train_rows, settings.max_rows)
        if not 2 <= settings.folds <= search_rows:
            raise InvalidParameterError(
                f"A validação cruzada precisa de entre 2 e {search_rows} folds; pediste "
                f"{settings.folds}."
            )
        baseline = SUGGESTED_MODELS.get(data.task_type)
        names = {model.info.name for agent in agents for model in agent.models}
        preview = AutoMLPlan(
            dataset_id=summary.id,
            dataset_name=summary.name,
            target=request.target,
            task_type=data.task_type,
            preparation=data.preparation,
            decisions=data.decisions,
            adopted=data.adopted,
            objective=_objective_option(objective.name),
            agents=tuple(_agent_card(agent) for agent in agents),
            skipped=skipped,
            baseline=baseline if baseline in names else None,
            train_rows=train_rows,
            test_rows=len(dataset.test),
            search_rows=search_rows,
            settings=settings,
        )
        return request, seeds, preview, objective, agents

    # --- a procura ---------------------------------------------------------------------

    def _search(self, plan: _Plan, control: TrainingControl | None) -> AutoMLResult:
        start = time.perf_counter()
        log, request, preview = plan.log, plan.request, plan.preview
        settings = request.settings
        try:
            log.enter(SearchPhase.DATA)
            for decision in preview.decisions:
                log.say(AgentRole.DATA, DATA_AGENT, f"{decision.title}. {decision.reason}")
            for text in _opening(preview, plan.objective):
                log.say(AgentRole.COORDINATOR, COORDINATOR, text)
            folds = self._datasets.folds(
                request.dataset,
                preview.preparation,
                plan.seeds.seed,
                settings.folds,
                settings.max_rows,
            )
            outcome = Coordinator(
                plan.objective,
                plan.agents,
                folds,
                budget=settings.budget,
                finalists=settings.finalists,
                model_seed=plan.seeds.seed_for(MODEL_SEED),
                search_seed=plan.seeds.seed_for(SEARCH_SEED),
                log=log,
                control=control,
                baseline=preview.baseline,
            ).search()
            log.enter(SearchPhase.FINAL)
            run_id, test = self._train_winner(plan, outcome)
        finally:
            log.enter(SearchPhase.DONE)
        baseline = next(
            (
                c.number
                for c in outcome.candidates
                if c.model == preview.baseline and not c.changes and not c.failed
            ),
            None,
        )
        result = AutoMLResult(
            search_id=plan.search_id,
            request=request,
            plan=preview,
            candidates=outcome.candidates,
            ranking=outcome.ranking,
            winner=None if outcome.winner is None else outcome.winner.number,
            baseline=baseline,
            messages=log.messages(),
            run_id=run_id,
            test=test,
            notices=(),
            duration_seconds=time.perf_counter() - start,
            stopped_early=outcome.stopped,
        )
        result = _with_notices(result, plan.objective)
        with self._lock:
            self._results[plan.search_id] = result
            self._results.move_to_end(plan.search_id)
            while len(self._results) > KEPT_SEARCHES:
                self._results.popitem(last=False)
        logger.info(
            "procura dos agentes concluída",
            extra={
                "search_id": plan.search_id,
                "candidates": len(outcome.candidates),
                "winner": None if outcome.winner is None else outcome.winner.model,
                "run_id": run_id,
            },
        )
        return result

    def _train_winner(
        self, plan: _Plan, outcome: SearchOutcome
    ) -> tuple[str | None, MetricResult | None]:
        """Treina o vencedor com o treino todo e avalia-o no teste (a única vez que se usa)."""
        winner, preview, objective = outcome.winner, plan.preview, plan.objective
        if winner is None:
            return None, None
        plan.log.say(
            AgentRole.COORDINATOR,
            COORDINATOR,
            f"A treinar o vencedor, {winner.model_display_name}, com o treino todo "
            f"({format_quantity(preview.train_rows, 'linha', 'linhas')}), e a avaliá-lo no teste "
            f"({format_quantity(preview.test_rows, 'linha', 'linhas')}), pela primeira e única "
            "vez.",
        )
        config = ExperimentConfig(
            dataset=plan.request.dataset,
            preparation=preview.preparation,
            model=ModelSpec(name=winner.model, hyperparameters=winner.hyperparameters),
            seed=plan.seeds.seed,
        )
        experiment = self._experiments.run(config)
        test = experiment.main_report.find(objective.name)
        self._name_run(experiment.run_id, f"Agentes · {winner.model_display_name}", preview)
        reading = _test_reading(winner, test, objective, sampled=preview.sampled)
        plan.log.say(AgentRole.COORDINATOR, COORDINATOR, reading)
        return experiment.run_id, test

    def _name_run(self, run_id: str, name: str, preview: AutoMLPlan) -> None:
        """Dá ao run do vencedor um nome que diz que veio dos agentes (no ecrã Experiências)."""
        if self._history is None:
            return
        try:
            self._history.rename(run_id, f"{name} · {preview.dataset_name} ({preview.target})")
        except MLLabError:  # o run treinou bem: um nome que falha não estraga a procura
            logger.exception("o run do vencedor ficou sem nome", extra={"run_id": run_id})

    def _job(self, search_id: str) -> TrainingJob[AutoMLResult]:
        job = self._jobs.find(search_id)
        if job is None:
            raise SearchNotFoundError(search_id)
        return job


def _objective_option(name: str) -> ObjectiveOption:
    metric = METRICS.get(name)
    explanation = METRIC_EXPLANATIONS.get(name)
    return ObjectiveOption(
        name=metric.name,
        label=metric.display_name,
        higher_is_better=metric.higher_is_better is not False,
        as_percent=metric.as_percent,
        needs_proba=metric.needs_proba,
        what=explanation.what if explanation is not None else "",
    )


def _agent_card(agent: ModelAgent) -> ModelAgentCard:
    models = tuple(
        AgentModel(
            name=model.info.name,
            display_name=model.info.display_name,
            searches=tuple(dimension.description for dimension in search_space(model).dimensions),
        )
        for model in agent.models
    )
    return ModelAgentCard(agent.family, agent.name, models)


def _opening(preview: AutoMLPlan, objective: Objective) -> list[str]:
    """O que o coordenador diz antes da triagem: o objetivo, os dados e os agentes."""
    texts = [
        f"Vou otimizar {objective.label} ({objective.direction}), medida com validação "
        f"cruzada em {preview.settings.folds} folds, só com as "
        f"{format_count(preview.train_rows)} linhas de treino. As "
        f"{format_count(preview.test_rows)} linhas de teste ficam guardadas: só o vencedor as "
        "vê, no fim."
    ]
    if preview.sampled:
        how = ", com a proporção de cada classe" if preview.task_type.is_classification else ""
        texts.append(
            f"O treino tem {format_count(preview.train_rows)} linhas: para a procura ser rápida, "
            f"a validação cruzada usa uma amostra de {format_count(preview.search_rows)}{how}. "
            "O vencedor treina no fim com todas."
        )
    agents = ", ".join(agent.family.label.lower() for agent in preview.agents)
    text = f"Chamei {format_quantity(len(preview.agents), 'agente', 'agentes')}: {agents}."
    if preview.skipped:
        text += f" Ficam de fora: {describe_skipped(preview.skipped)}."
    texts.append(text)
    return texts


def _test_reading(
    winner: Candidate, test: MetricResult | None, objective: Objective, *, sampled: bool
) -> str:
    """O que o coordenador diz do teste do vencedor, comparado com a validação cruzada."""
    if test is None or test.value is None or winner.mean is None:
        return "O teste não deu um valor para o objetivo (faltava alguma classe no teste)."
    expected = f"{objective.written(winner.mean)} ± {objective.written(winner.std)}"
    text = f"No teste: {test.formatted}. Na validação cruzada esperava-se {expected}."
    spread = winner.std or 0.0
    if abs(test.value - winner.mean) <= spread:
        return f"{text} Dentro do previsto: a estimativa da validação cruzada era honesta."
    if objective.is_better(test.value, winner.mean):
        extra = (
            " O modelo final treinou com todas as linhas, e a procura só com uma amostra: com "
            "mais dados, é normal sair melhor."
            if sampled
            else " O teste também é uma amostra, e também tem sorte."
        )
        return f"{text} Acima do previsto.{extra}"
    return (
        f"{text} Abaixo do previsto: entre muitos candidatos, o vencedor também teve alguma "
        "sorte nos folds, e o teste corrige-o (o vencedor é sempre um pouco otimista)."
    )


def _with_notices(result: AutoMLResult, objective: Objective) -> AutoMLResult:
    """O resultado com a sua leitura: a referência, o empate, os candidatos que falharam."""
    notices: list[Notice] = []
    winner = result.winner_candidate
    ranked = [result.candidate(number) for number in result.ranking]
    if result.stopped_early:
        notices.append(
            Notice.info(
                "stopped_early",
                f"A procura parou a pedido ao fim de "
                f"{format_quantity(len(result.candidates), 'candidato', 'candidatos')}: o "
                "vencedor é o melhor até aí.",
            )
        )
    if winner is not None and winner.mean is not None and result.baseline is not None:
        reference = result.candidate(result.baseline)
        assert reference.mean is not None  # a referência conta só se foi medida
        if reference.number == winner.number:
            notices.append(
                Notice.info(
                    "baseline_won",
                    f"A referência ({reference.model_display_name}, com os valores por omissão) "
                    "ganhou: nenhum modelo mais complexo a bateu. Às vezes, o mais simples chega.",
                )
            )
        else:
            gap = objective.gap(winner.mean, reference.mean)
            notices.append(
                Notice.info(
                    "baseline_beaten",
                    f"O vencedor bate a referência ({reference.model_display_name}, "
                    f"{objective.written(reference.mean)}) por {gap}, na validação cruzada.",
                )
            )
    if winner is not None and len(ranked) > 1 and winner.mean is not None:
        second = ranked[1]
        if second.mean is not None and abs(winner.mean - second.mean) < (winner.std or 0.0):
            notices.append(
                Notice.info(
                    "close_race",
                    f"O segundo ({second.model_display_name}"
                    f"{', com ' + describe_changes(second.changes) if second.changes else ''}) "
                    f"fica a {objective.gap(winner.mean, second.mean)}, menos do que o desvio "
                    "padrão entre folds: na prática, empatam.",
                )
            )
    failed = [candidate for candidate in result.candidates if candidate.failed]
    if failed:
        names = ", ".join(sorted({candidate.model_display_name for candidate in failed}))
        notices.append(
            Notice.warning(
                "failed_candidates",
                f"{format_quantity(len(failed), 'candidato falhou', 'candidatos falharam')} na "
                f"validação cruzada ({names}): ficam no quadro, com a razão.",
            )
        )
    if winner is None:
        notices.append(
            Notice.warning(
                "no_winner",
                "Nenhum candidato chegou ao fim da validação cruzada: não há vencedor. Vê as "
                "razões no quadro, ou experimenta outras famílias de modelos.",
            )
        )
    return replace(result, notices=tuple(notices))


def _experiment(result: AutoMLResult, candidate: Candidate) -> ExperimentConfig:
    """A experiência de um candidato: os dados e a preparação da procura, e a seed dela."""
    return ExperimentConfig(
        dataset=result.request.dataset,
        preparation=result.plan.preparation,
        model=ModelSpec(name=candidate.model, hyperparameters=candidate.hyperparameters),
        seed=result.seed,
    )
