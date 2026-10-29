"""Serviço de reinforcement learning: treinar um agente num ambiente e mostrar o que aprendeu.

É o ecrã Reinforcement learning, e segue o mesmo fluxo do serviço de experiências:

1. **o plano** (falha logo, antes de qualquer evento): a seed raiz e as seeds de cada parte
   (ADR 0006), o ambiente e o agente com os parâmetros validados;
2. **o treino**: o agente joga os episódios com um contexto de treino que publica cada
   episódio e cada snapshot no ``EventBus``, e por isso pausa e pára como qualquer modelo;
3. **a avaliação**: os últimos episódios do treino e episódios novos com a política
   aprendida, sem explorar (``core/rl/evaluation.py``);
4. **o fim**: o resultado e o anúncio ``RunFinished``. O histórico regista-o como regista os
   outros runs (ADR 0020).

``run`` faz tudo e devolve o resultado; ``start`` treina em segundo plano, e ``job``,
``updates``, ``pause``, ``resume``, ``cancel`` e ``result`` servem quem só sabe o
identificador do run (a API, ADR 0022). ``lesson`` é a animação que abre o ecrã: um
Q-learning pequeno, passo a passo, que não é um run (ADR 0037).
"""

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from ml_lab.config import Settings, get_settings
from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.report import EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import (
    HyperparameterError,
    InvalidParameterError,
    RunNotInMemoryError,
)
from ml_lab.core.foundation.events import (
    EvaluationCompleted,
    EventBus,
    RunFailed,
    RunFinished,
    RunStarted,
)
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Implementation
from ml_lab.core.rl.agents.base import AGENTS, Agent
from ml_lab.core.rl.environments.base import ENVIRONMENTS, Environment, GridLayout
from ml_lab.core.rl.evaluation import (
    EVALUATION_EPISODES,
    episodes_report,
    play,
    reading,
    training_outcomes,
)
from ml_lab.core.rl.lesson import ReinforcementLesson, lesson
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.schemas.reinforcement import ReinforcementConfig
from ml_lab.core.schemas.serialization import internal, to_json
from ml_lab.core.training.context import EventTrainingContext, SnapshotSampler
from ml_lab.core.training.control import TrainingControl
from ml_lab.core.training.jobs import JobRegistry, JobState, TrainingJob
from ml_lab.core.training.recorder import SnapshotRecorder, StepRecord
from ml_lab.core.visualization_data.policy import (
    PolicyFrame,
    PolicyTrace,
    RewardCurve,
    policy_frame,
    policy_trace,
    reward_curve,
)
from ml_lab.formatting import format_count
from ml_lab.logging import get_logger

logger = get_logger(__name__)

AGENT_SEED: Final = "agent"
EVALUATION_SEED: Final = "evaluation"

KEPT_RUNS: Final = 5
"""Quantos runs ficam em memória (o agente treinado), como no serviço de experiências."""

DOCS_FOLDER: Final = Path("docs") / "algorithms"


@dataclass(frozen=True, slots=True)
class EnvironmentCard:
    """Um ambiente, pronto a mostrar.

    Attributes:
        name: Identificador no registo.
        display_name: Nome a mostrar.
        summary: O que é, numa frase.
        parameters_schema: JSON Schema dos parâmetros (de onde se gera o formulário).
        defaults: Os valores por omissão dos parâmetros.
    """

    name: str
    display_name: str
    summary: str
    parameters_schema: dict[str, Any]
    defaults: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentCard:
    """Um agente, pronto a mostrar.

    Attributes:
        name: Identificador no registo.
        display_name: Nome a mostrar.
        summary: O que faz, numa frase.
        implementation: Didático (NumPy) ou PyTorch.
        hyperparameters_schema: JSON Schema dos hiperparâmetros.
        defaults: Os valores por omissão dos hiperparâmetros.
        doc: O texto da ficha didática, em Markdown (vazio se o ficheiro faltar).
    """

    name: str
    display_name: str
    summary: str
    implementation: Implementation
    hyperparameters_schema: dict[str, Any]
    defaults: dict[str, Any]
    doc: str


@dataclass(frozen=True, slots=True)
class ReinforcementResult:
    """O resultado de um treino de reinforcement learning, pronto a mostrar.

    Tem os mesmos nomes que o ``ExperimentResult`` naquilo que o histórico e a comparação de
    runs usam (``reports``, ``history``, ``main_report``…): o ambiente faz o papel do
    dataset, e o agente o do modelo.

    Attributes:
        run_id: Identificador do run.
        config: A configuração, com a seed já resolvida: repete o treino tal e qual.
        environment_name: O ambiente, com o que o distingue ("Grelha · penhasco").
        agent_name: O agente, no registo.
        agent_display_name: O nome a mostrar do agente.
        hyperparameters: Os hiperparâmetros usados, completos.
        environment_parameters: Os parâmetros do ambiente, completos.
        reports: O treino (os últimos episódios, a explorar) e o teste (a política aprendida).
        notices: Todos os avisos: os do treino e os da leitura do treino contra o teste.
        duration_seconds: Quanto tempo demorou o treino e a avaliação.
        layout: A grelha do ambiente.
        learned: O que o agente sabe no fim: o valor e a ação de cada célula e o caminho.
        rewards: A recompensa de cada episódio, com as médias móveis.
        history: Cada episódio do treino (a loss é o erro médio de diferença temporal).
        stopped_early: Se o treino foi interrompido a pedido.
        training_notices: Só os avisos do treino.
        trace: O que o agente sabia em cada snapshot (pesa mais: só se guarda a pedido).
        task_type: Sempre o controlo.
        main_partition: A parte que resume o run: o teste (a política aprendida).
    """

    run_id: str
    config: ReinforcementConfig
    environment_name: str
    agent_name: str
    agent_display_name: str
    hyperparameters: dict[str, Any]
    environment_parameters: dict[str, Any]
    reports: Mapping[Partition, EvaluationReport]
    notices: tuple[Notice, ...]
    duration_seconds: float
    layout: GridLayout
    learned: PolicyFrame
    rewards: RewardCurve
    history: tuple[StepRecord, ...] = ()
    stopped_early: bool = False
    training_notices: tuple[Notice, ...] = ()
    trace: PolicyTrace | None = None
    task_type: TaskType = TaskType.CONTROL
    main_partition: Partition = Partition.TEST

    @property
    def seed(self) -> int:
        """A seed raiz usada."""
        assert self.config.seed is not None  # o serviço resolve-a sempre antes de criar o resultado
        return self.config.seed

    @property
    def steps(self) -> int:
        """Quantos episódios o treino deu."""
        return self.history[-1].step if self.history else 0

    @property
    def main_report(self) -> EvaluationReport:
        """O relatório que resume o run: o da política aprendida."""
        return self.reports[self.main_partition]

    @property
    def primary(self) -> MetricResult | None:
        """A taxa de sucesso da política aprendida."""
        name = PRIMARY_METRIC.get(self.task_type)
        return None if name is None else self.main_report.find(name)

    @property
    def dataset_name(self) -> str:
        """O ambiente (o histórico guarda-o onde guarda o dataset dos outros runs)."""
        return self.environment_name

    @property
    def model_name(self) -> str:
        """O agente (o histórico guarda-o onde guarda o modelo dos outros runs)."""
        return self.agent_name

    @property
    def model_display_name(self) -> str:
        """O nome a mostrar do agente."""
        return self.agent_display_name


@dataclass(frozen=True, slots=True)
class ReinforcementUpdate:
    """O que mudou num treino em segundo plano desde a última pergunta (o que a API transmite).

    Attributes:
        state: O retrato do treino agora.
        steps: Os episódios novos (depois de ``after_step``), pela ordem.
        preview: O que o agente sabia no último snapshot, se houver um snapshot novo.
        layout: A grelha onde se desenha o ``preview`` (só vem com ele).
    """

    state: JobState
    steps: tuple[StepRecord, ...] = ()
    preview: PolicyFrame | None = None
    layout: GridLayout | None = None

    @property
    def snapshot_step(self) -> int | None:
        """O episódio do snapshot novo, se houver."""
        return self.preview.episode if self.preview is not None else None


@dataclass(frozen=True, slots=True, eq=False)
class ReinforcementArtifacts:
    """O agente treinado, o ambiente e o resultado de um run (para o guardar)."""

    agent: Agent[Any]
    environment: Environment[Any]
    config: ReinforcementConfig
    result: ReinforcementResult


@dataclass(frozen=True, slots=True, eq=False)
class _Plan:
    """O que fica decidido antes de treinar."""

    run_id: str
    config: ReinforcementConfig
    environment: Environment[Any]
    agent: Agent[Any]
    evaluation_seed: int
    sampler: SnapshotSampler
    recorder: SnapshotRecorder


class ReinforcementService:
    """Treina agentes.

    Args:
        settings: As definições (a raiz do projeto, onde estão as fichas didáticas).
        bus: Para onde publicar os eventos dos runs; por omissão, um novo.
    """

    def __init__(self, settings: Settings | None = None, *, bus: EventBus | None = None) -> None:
        self._docs = (settings or get_settings()).home / DOCS_FOLDER
        self._bus = bus or EventBus()
        self._runs: OrderedDict[str, ReinforcementArtifacts] = OrderedDict()
        self._active: dict[str, _Plan] = {}
        self._jobs: JobRegistry[ReinforcementResult] = JobRegistry(KEPT_RUNS)
        self._lock = threading.Lock()
        self._lesson: ReinforcementLesson | None = None

    @property
    def bus(self) -> EventBus:
        """O ``EventBus`` onde os runs anunciam o que acontece."""
        return self._bus

    def environments(self) -> list[EnvironmentCard]:
        """Os ambientes, com o formulário dos parâmetros de cada um."""
        cards = []
        for _, environment in ENVIRONMENTS.items():
            info = environment.info
            cards.append(
                EnvironmentCard(
                    name=info.name,
                    display_name=info.display_name,
                    summary=info.summary,
                    parameters_schema=info.parameters.form_schema(),
                    defaults=info.parameters().model_dump(mode="json"),
                )
            )
        return cards

    def agents(self) -> list[AgentCard]:
        """Os agentes, com o formulário dos hiperparâmetros e a ficha didática de cada um."""
        cards = []
        for _, agent in AGENTS.items():
            info = agent.info
            doc_path = self._docs / info.doc
            cards.append(
                AgentCard(
                    name=info.name,
                    display_name=info.display_name,
                    summary=info.summary,
                    implementation=info.implementation,
                    hyperparameters_schema=info.hyperparameters.form_schema(),
                    defaults=info.hyperparameters().model_dump(mode="json"),
                    doc=doc_path.read_text(encoding="utf-8") if doc_path.is_file() else "",
                )
            )
        return sorted(cards, key=lambda card: card.name != "q_learning")  # o didático primeiro

    def lesson(self) -> ReinforcementLesson:
        """A lição do ecrã: um Q-learning no penhasco, visto passo a passo.

        É o que a animação "Como um agente aprende" mostra (``core/rl/lesson.py``): alguns
        episódios inteiros, com cada passo (o estado, a ação, se explorou, a recompensa, a
        surpresa e o valor novo) e uma frase sobre cada episódio. Não é um run: não vai para o
        histórico. É sempre a mesma (a seed é fixa), por isso calcula-se uma vez e fica guardada.
        """
        if self._lesson is None:
            self._lesson = lesson()
        return self._lesson

    def run(self, config: ReinforcementConfig) -> ReinforcementResult:
        """Treina o agente descrito por ``config`` e espera pelo resultado.

        Raises:
            ComponentNotFoundError: Se o ambiente ou o agente não existirem.
            InvalidParameterError: Se os parâmetros do ambiente não forem válidos.
            HyperparameterError: Se os hiperparâmetros do agente não forem válidos.
        """
        return self._train(self._plan(config), None)

    def start(self, config: ReinforcementConfig) -> TrainingJob[ReinforcementResult]:
        """Valida o treino e corre-o em segundo plano.

        Raises:
            ComponentNotFoundError, InvalidParameterError, HyperparameterError: Os mesmos
                problemas de ``run``, logo aqui.
        """
        plan = self._plan(config)
        with self._lock:
            self._active[plan.run_id] = plan
        job: TrainingJob[ReinforcementResult] = TrainingJob(
            plan.run_id,
            lambda control: self._train(plan, control),
            plan.recorder,
            planned_steps=plan.agent.hyperparameters.episodes,
        )
        return self._jobs.add(job.start())

    def job(self, run_id: str) -> JobState:
        """O estado de um treino lançado com ``start``.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        return self._jobs.get(run_id).state()

    def pause(self, run_id: str) -> JobState:
        """Pede uma pausa no fim do episódio atual.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        job = self._jobs.get(run_id)
        job.pause()
        return job.state()

    def resume(self, run_id: str) -> JobState:
        """Retoma um treino em pausa.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        job = self._jobs.get(run_id)
        job.resume()
        return job.state()

    def cancel(self, run_id: str) -> JobState:
        """Pede ao treino que pare no fim do episódio atual (o agente fica com o que aprendeu).

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        job = self._jobs.get(run_id)
        job.cancel()
        return job.state()

    def updates(
        self, run_id: str, after_step: int = 0, after_snapshot: int = 0
    ) -> ReinforcementUpdate:
        """O que mudou num treino desde a última pergunta.

        Args:
            run_id: O run.
            after_step: O último episódio que quem pergunta já tem.
            after_snapshot: O episódio do último snapshot que quem pergunta já tem.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        job = self._jobs.get(run_id)
        state = job.state()  # primeiro: se diz que acabou, os episódios lidos a seguir estão todos
        steps = tuple(record for record in job.recorder.steps() if record.step > after_step)
        snapshots = job.recorder.snapshots()
        with self._lock:
            plan = self._active.get(run_id)
        if plan is None or not snapshots or snapshots[-1].step <= after_snapshot:
            return ReinforcementUpdate(state, steps)
        last, layout = snapshots[-1], plan.environment.layout()
        return ReinforcementUpdate(
            state, steps, policy_frame(last.step, last.state, layout), layout
        )

    def result(self, run_id: str) -> ReinforcementResult:
        """O resultado de um run desta sessão.

        Raises:
            TrainingInProgressError: Se o run ainda estiver a treinar.
            TrainingFailedError: Se o treino falhou.
            RunNotInMemoryError: Se o run já não estiver em memória.
        """
        if self._jobs.find(run_id) is not None:
            return self._jobs.result(run_id)
        return self.artifacts(run_id).result

    @internal
    def artifacts(self, run_id: str) -> ReinforcementArtifacts:
        """O agente treinado e o resultado de um run recente (para o guardar).

        Raises:
            RunNotInMemoryError: Se o run já não estiver em memória.
        """
        with self._lock:
            artifacts = self._runs.get(run_id)
        if artifacts is None:
            raise RunNotInMemoryError(run_id)
        return artifacts

    def _plan(self, config: ReinforcementConfig) -> _Plan:
        seeds = SeedManager.create(config.seed)
        config = config.model_copy(update={"seed": seeds.seed})
        environment_class = ENVIRONMENTS.get(config.environment.name)
        environment = environment_class(
            _environment_parameters(environment_class, config.environment.parameters)
        )
        agent_class = AGENTS.get(config.agent.name)
        info = agent_class.info
        hyperparameters = info.hyperparameters.parse(config.agent.hyperparameters, model=info.name)
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        sampler = SnapshotSampler()
        return _Plan(
            run_id=run_id,
            config=config,
            environment=environment,
            agent=agent_class(hyperparameters, seed=seeds.seed_for(AGENT_SEED)),
            evaluation_seed=seeds.seed_for(EVALUATION_SEED),
            sampler=sampler,
            recorder=SnapshotRecorder(run_id, sampler),
        )

    def _train(self, plan: _Plan, control: TrainingControl | None) -> ReinforcementResult:
        environment, agent = plan.environment, plan.agent
        plan.recorder.attach(self._bus)
        self._bus.publish(
            RunStarted(
                run_id=plan.run_id,
                config=plan.config.model_dump(mode="json"),
                n_samples=0,  # não há dataset: o agente aprende a jogar
                n_features=environment.n_states,
            )
        )
        start = time.perf_counter()
        try:
            context = EventTrainingContext(
                self._bus, plan.run_id, sampler=plan.sampler, control=control
            )
            agent.train(environment, context)
            # A fotografia do fim, mesmo que o último episódio não calhasse na amostragem.
            if context.last_step and context.last_snapshot_step != context.last_step:
                context.report_snapshot(context.last_step, agent.snapshot_state(environment))
            history = plan.recorder.steps()
            train = episodes_report(training_outcomes(history), Partition.TRAIN)
            test = episodes_report(
                play(agent, environment, EVALUATION_EPISODES, plan.evaluation_seed),
                Partition.TEST,
            )
            self._bus.publish(
                EvaluationCompleted(
                    run_id=plan.run_id, train_report=to_json(train), test_report=to_json(test)
                )
            )
            duration = time.perf_counter() - start
            stopped = control is not None and control.stop_requested
            result = self._result(plan, history, {Partition.TRAIN: train, Partition.TEST: test})
            result = _finished(result, duration, stopped)
        except Exception as error:
            self._bus.publish(
                RunFailed(run_id=plan.run_id, message=str(error), error_type=type(error).__name__)
            )
            raise
        finally:
            plan.recorder.detach(self._bus)
            with self._lock:
                self._active.pop(plan.run_id, None)
        self._keep(ReinforcementArtifacts(agent, environment, plan.config, result))
        primary = result.primary
        summary: dict[str, Any] = {"steps": result.steps}
        if primary is not None:
            summary[primary.name] = primary.value
        self._bus.publish(
            RunFinished(run_id=plan.run_id, duration_seconds=duration, summary=summary)
        )
        logger.info(
            "treino de reinforcement learning concluído",
            extra={"run_id": plan.run_id, "agent": agent.info.name, **summary},
        )
        return result

    def _result(
        self,
        plan: _Plan,
        history: tuple[StepRecord, ...],
        reports: dict[Partition, EvaluationReport],
    ) -> ReinforcementResult:
        environment, agent = plan.environment, plan.agent
        layout = environment.layout()
        slip = float(environment.parameters.model_dump().get("slip", 0.0))
        training = list(agent.notices)
        return ReinforcementResult(
            run_id=plan.run_id,
            config=plan.config,
            environment_name=environment.title(),
            agent_name=agent.info.name,
            agent_display_name=agent.info.display_name,
            hyperparameters=agent.hyperparameters.model_dump(mode="json"),
            environment_parameters=environment.parameters.model_dump(mode="json"),
            reports=reports,
            notices=(*training, *reading(reports[Partition.TRAIN], reports[Partition.TEST], slip)),
            duration_seconds=0.0,
            layout=layout,
            learned=policy_frame(len(history), agent.snapshot_state(environment), layout),
            rewards=reward_curve(history),
            history=history,
            training_notices=tuple(training),
            trace=policy_trace(
                layout, [(record.step, record.state) for record in plan.recorder.snapshots()]
            ),
        )

    def _keep(self, artifacts: ReinforcementArtifacts) -> None:
        with self._lock:
            self._runs[artifacts.result.run_id] = artifacts
            self._runs.move_to_end(artifacts.result.run_id)
            while len(self._runs) > KEPT_RUNS:
                self._runs.popitem(last=False)


def _environment_parameters(
    environment_class: type[Environment[Any]], values: Mapping[str, Any]
) -> Hyperparameters:
    """Os parâmetros do ambiente, validados (com o erro dito como de um ambiente)."""
    name = environment_class.info.name
    try:
        return environment_class.info.parameters.parse(values, model=name)
    except HyperparameterError as error:
        raise InvalidParameterError(
            f"Parâmetros inválidos para o ambiente '{name}': {' '.join(error.problems)}"
        ) from error


def _finished(result: ReinforcementResult, duration: float, stopped: bool) -> ReinforcementResult:
    """O resultado com a duração e, se o treino parou a pedido, o aviso disso."""
    if not stopped:
        return replace(result, duration_seconds=duration)
    notice = Notice.info(
        "stopped_early",
        f"Treino parado a pedido no episódio {format_count(result.steps)}: o agente ficou com "
        "o que tinha aprendido até aí.",
    )
    return replace(
        result,
        duration_seconds=duration,
        stopped_early=True,
        notices=(notice, *result.notices),
        training_notices=(notice, *result.training_notices),
    )
