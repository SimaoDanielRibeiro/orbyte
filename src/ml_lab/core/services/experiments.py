"""Serviço de experiências: preparar os dados, treinar, avaliar e anunciar o run.

É o fluxo completo da secção "Fluxo de uma experiência" do plano:

1. resolve a seed raiz e as seeds de cada componente (ADR 0006);
2. prepara os dados com a configuração (a camada de dados devolve o ``Dataset``);
3. vai buscar o modelo ao registo e valida os hiperparâmetros;
4. anuncia o início do run e manda o Trainer treinar: cada passo e cada snapshot (com
   as previsões na grelha da sonda) vão para o ``EventBus`` e para o gravador;
5. avalia cada parte dos dados e compara treino e teste (o alerta de overfitting);
6. monta a animação do treino e anuncia o fim.

Os passos 1 a 3 são o plano do run: se a configuração tiver um problema, falha logo,
antes de qualquer evento. ``run`` faz tudo e devolve o resultado; ``start`` valida o
plano e treina em segundo plano, num ``TrainingJob`` que se pode pausar, retomar e
parar (ADR 0014).

Quem só sabe o identificador do run (a API) acompanha o treino com ``job`` e
``updates``, controla-o com ``pause``, ``resume`` e ``cancel``, e pede o resultado com
``result`` (ADR 0022).
"""

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.evaluation.base import PRIMARY_METRIC, evaluator_for
from ml_lab.core.evaluation.diagnostics import compare_train_and_test
from ml_lab.core.evaluation.report import EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import ModelError, RunNotInMemoryError
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
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.schemas.experiment import ExperimentConfig
from ml_lab.core.schemas.serialization import internal, to_json
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.training.context import SnapshotSampler
from ml_lab.core.training.control import TrainingControl
from ml_lab.core.training.jobs import JobRegistry, JobState, TrainingJob
from ml_lab.core.training.recorder import SnapshotRecorder, StepRecord
from ml_lab.core.training.trainer import Trainer, TrainingOutcome
from ml_lab.core.visualization_data.animation import TrainingAnimation, build_animation
from ml_lab.core.visualization_data.embedding import Embedding, embedding_map
from ml_lab.core.visualization_data.grid import ProbeGrid, probe_grid
from ml_lab.core.visualization_data.learned import LearnedView, learned_view
from ml_lab.core.visualization_data.network import NetworkTrace, network_trace
from ml_lab.formatting import format_count
from ml_lab.logging import get_logger

logger = get_logger(__name__)

MODEL_SEED: Final = "model"
"""Nome do componente do modelo: a sua seed é ``seeds.seed_for("model")``."""

KEPT_RUNS: Final = 5
"""Quantos runs ficam em memória (modelo e dados) para análises depois do treino."""


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """O resultado de uma experiência, pronto a mostrar no ecrã Treinar e avaliar.

    Attributes:
        run_id: Identificador do run (é o que aparece nos eventos).
        config: A configuração, com a seed já resolvida: repete o run tal e qual.
        dataset_name: O nome do dataset.
        task_type: O tipo de tarefa.
        model_name: O modelo, no registo.
        model_display_name: O nome a mostrar do modelo.
        hyperparameters: Os hiperparâmetros usados, completos.
        feature_names: As features que o modelo recebeu.
        class_labels: Na classificação, as classes.
        reports: A avaliação de cada parte dos dados.
        model_state: O que o modelo aprendeu (coeficientes...).
        notices: Todos os avisos: os do treino e os dos diagnósticos.
        duration_seconds: Quanto tempo demorou o treino e a avaliação.
        history: Cada passo do treino, com a loss (vazio nos modelos que treinam de uma vez).
        stopped_early: Se o treino foi interrompido a pedido.
        animation: A fronteira (ou a reta) em cada snapshot, para rever o treino.
        training_notices: Só os avisos do treino (não convergiu, explodiu, foi parado).
        embedding: Na redução de dimensionalidade, o mapa dos pontos de treino.
        learned: O que o modelo aprendeu, pronto a mostrar (pesos, regras, centros…).
        main_partition: A parte dos dados cujo relatório resume o run: o teste, ou o
            treino num modelo que só conhece os pontos com que treinou (o t-SNE).
        network: Numa rede neuronal, o que cada camada fez em cada snapshot (as ativações
            dos neurónios e os gradientes).
    """

    run_id: str
    config: ExperimentConfig
    dataset_name: str
    task_type: TaskType
    model_name: str
    model_display_name: str
    hyperparameters: dict[str, Any]
    feature_names: tuple[str, ...]
    class_labels: tuple[str, ...]
    reports: Mapping[Partition, EvaluationReport]
    model_state: dict[str, Any]
    notices: tuple[Notice, ...]
    duration_seconds: float
    history: tuple[StepRecord, ...] = ()
    stopped_early: bool = False
    animation: TrainingAnimation | None = None
    training_notices: tuple[Notice, ...] = ()
    embedding: Embedding | None = None
    learned: LearnedView = field(default_factory=LearnedView)
    main_partition: Partition = Partition.TEST
    network: NetworkTrace | None = None

    @property
    def seed(self) -> int:
        """A seed raiz usada."""
        assert self.config.seed is not None  # o serviço resolve-a sempre antes de criar o resultado
        return self.config.seed

    @property
    def steps(self) -> int:
        """Quantos passos o treino deu (0 nos modelos que treinam de uma vez)."""
        return self.history[-1].step if self.history else 0

    @property
    def main_report(self) -> EvaluationReport:
        """O relatório que resume o run (o da parte ``main_partition``)."""
        return self.reports[self.main_partition]

    @property
    def primary(self) -> MetricResult | None:
        """A métrica principal no relatório principal (por exemplo, a exatidão no teste)."""
        name = PRIMARY_METRIC.get(self.task_type)
        return None if name is None else self.main_report.find(name)


@dataclass(frozen=True, slots=True)
class TrainingUpdate:
    """O que mudou num treino em segundo plano desde a última pergunta.

    Quem acompanha o treino diz o que já tem (os passos até ``after_step`` e o snapshot do
    passo ``after_snapshot``) e recebe só o que é novo. É o que a API transmite, tanto a quem
    pergunta de vez em quando como pelo WebSocket.

    Attributes:
        state: O retrato do treino agora.
        steps: Os passos novos, pela ordem.
        preview: A fronteira no último snapshot, se houver um snapshot novo.
    """

    state: JobState
    steps: tuple[StepRecord, ...] = ()
    preview: TrainingAnimation | None = None

    @property
    def snapshot_step(self) -> int | None:
        """O passo do snapshot novo, se houver."""
        return self.preview.frames[-1].step if self.preview is not None else None


@dataclass(frozen=True, slots=True, eq=False)
class RunArtifacts:
    """O modelo treinado, os dados e o resultado de um run, para o que vem depois do treino."""

    model: LabModel[Any]
    dataset: Dataset
    config: ExperimentConfig
    result: ExperimentResult


@dataclass(frozen=True, slots=True, eq=False)
class _Plan:
    """O que fica decidido antes de treinar: o run, a configuração, o modelo e os dados."""

    run_id: str
    config: ExperimentConfig
    model: LabModel[Any]
    dataset: Dataset
    grid: ProbeGrid | None
    sampler: SnapshotSampler
    recorder: SnapshotRecorder

    @property
    def planned_steps(self) -> int | None:
        """O máximo de passos, se o modelo declarar o hiperparâmetro que o limita."""
        field = self.model.info.step_limit
        return None if field is None else int(getattr(self.model.hyperparameters, field))


class ExperimentService:
    """Corre experiências.

    Args:
        datasets: O serviço de datasets (prepara os dados e guarda-os em memória).
        bus: Para onde publicar os eventos dos runs; por omissão, um novo.
    """

    def __init__(self, datasets: DatasetService, *, bus: EventBus | None = None) -> None:
        self._datasets = datasets
        self._bus = bus or EventBus()
        self._runs: OrderedDict[str, RunArtifacts] = OrderedDict()
        self._active: dict[str, _Plan] = {}
        self._jobs: JobRegistry[ExperimentResult] = JobRegistry(KEPT_RUNS)
        self._lock = threading.Lock()

    @property
    def bus(self) -> EventBus:
        """O ``EventBus`` onde os runs anunciam o que acontece."""
        return self._bus

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Corre a experiência descrita por ``config`` e espera pelo resultado.

        Raises:
            ComponentNotFoundError: Se o modelo não existir.
            HyperparameterError: Se os hiperparâmetros não forem válidos.
            ModelError: Se o modelo não resolver o tipo de tarefa dos dados.
            DataError: Se os dados não servirem para a preparação pedida.
        """
        return self._train(self._plan(config), None)

    def start(self, config: ExperimentConfig) -> TrainingJob[ExperimentResult]:
        """Valida a experiência e treina-a em segundo plano.

        Returns:
            O trabalho: dá o progresso, aceita pausas e a ordem de parar, e no fim tem o
            resultado (ou o erro).

        Raises:
            ComponentNotFoundError, HyperparameterError, ModelError, DataError: Os mesmos
                problemas de ``run`` que se detetam antes de treinar, logo aqui.
        """
        plan = self._plan(config)
        with self._lock:
            self._active[plan.run_id] = plan
        job: TrainingJob[ExperimentResult] = TrainingJob(
            plan.run_id,
            lambda control: self._train(plan, control),
            plan.recorder,
            planned_steps=plan.planned_steps,
        )
        return self._jobs.add(job.start())

    def job(self, run_id: str) -> JobState:
        """O estado de um treino lançado com ``start``.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        return self._jobs.get(run_id).state()

    def pause(self, run_id: str) -> JobState:
        """Pede uma pausa no fim do passo atual.

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
        """Pede ao treino que pare no fim do passo atual (o modelo fica com o que aprendeu).

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        job = self._jobs.get(run_id)
        job.cancel()
        return job.state()

    def updates(self, run_id: str, after_step: int = 0, after_snapshot: int = 0) -> TrainingUpdate:
        """O que mudou num treino desde a última pergunta.

        Args:
            run_id: O run.
            after_step: O último passo que quem pergunta já tem.
            after_snapshot: O passo do último snapshot que quem pergunta já tem.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        job = self._jobs.get(run_id)
        # O estado primeiro: se diz que o treino acabou, os passos lidos a seguir já estão todos.
        state = job.state()
        steps = tuple(record for record in job.recorder.steps() if record.step > after_step)
        snapshots = job.recorder.snapshots()
        # A fronteira custa a calcular (a grelha inteira): só quando há um snapshot novo.
        has_new = bool(snapshots) and snapshots[-1].step > after_snapshot
        return TrainingUpdate(state, steps, self.preview(run_id) if has_new else None)

    def result(self, run_id: str) -> ExperimentResult:
        """O resultado de um run desta sessão.

        Raises:
            TrainingInProgressError: Se o run ainda estiver a treinar.
            TrainingFailedError: Se o treino falhou.
            RunNotInMemoryError: Se o run já não estiver em memória.
        """
        if self._jobs.find(run_id) is not None:
            return self._jobs.result(run_id)
        return self.artifacts(run_id).result

    def preview(self, run_id: str) -> TrainingAnimation | None:
        """A fronteira (ou a reta) no último snapshot de um treino que ainda decorre.

        Returns:
            Uma animação com uma só fotografia, ou ``None`` se o run não estiver a treinar,
            não tiver grelha ou ainda não tiver tirado nenhum snapshot.
        """
        with self._lock:
            plan = self._active.get(run_id)
        if plan is None or plan.grid is None:
            return None
        snapshots = plan.recorder.snapshots()
        if not snapshots:
            return None
        last = snapshots[-1]
        loss = next((s.loss for s in reversed(plan.recorder.steps()) if s.step == last.step), None)
        return build_animation(
            plan.dataset,
            plan.grid,
            [(last.step, last.state)],
            losses={last.step: loss},
            classes=plan.model.classes(),
        )

    @internal
    def artifacts(self, run_id: str) -> RunArtifacts:
        """O modelo e os dados de um run recente (para o serviço de análise).

        Raises:
            RunNotInMemoryError: Se o run já não estiver em memória.
        """
        with self._lock:
            artifacts = self._runs.get(run_id)
        if artifacts is None:
            raise RunNotInMemoryError(run_id)
        return artifacts

    @internal
    def remember(self, artifacts: RunArtifacts) -> None:
        """Volta a pôr em memória um run (um run guardado que se reabre do histórico)."""
        self._keep(artifacts.result.run_id, artifacts)

    def _plan(self, config: ExperimentConfig) -> _Plan:
        seeds = SeedManager.create(config.seed)
        config = config.model_copy(update={"seed": seeds.seed})
        model_class = MODELS.get(config.model.name)
        info = model_class.info
        hyperparameters = info.hyperparameters.parse(config.model.hyperparameters, model=info.name)
        dataset = self._datasets.prepared(config.dataset, config.preparation, seeds.seed).dataset
        if dataset.task_type not in info.task_types:
            raise ModelError(
                f"'{info.display_name}' doesn't solve {dataset.task_type.label.lower()}. Choose a "
                "model from the catalogue filtered by this task."
            )
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        sampler = SnapshotSampler()
        return _Plan(
            run_id=run_id,
            config=config,
            model=model_class(hyperparameters, seed=seeds.seed_for(MODEL_SEED)),
            dataset=dataset,
            # A redução de dimensionalidade não tem campo para desenhar: o resultado é o mapa.
            grid=None
            if dataset.task_type is TaskType.DIMENSIONALITY_REDUCTION
            else probe_grid(dataset),
            sampler=sampler,
            recorder=SnapshotRecorder(run_id, sampler),
        )

    def _train(self, plan: _Plan, control: TrainingControl | None) -> ExperimentResult:
        dataset = plan.dataset
        plan.recorder.attach(self._bus)
        self._bus.publish(
            RunStarted(
                run_id=plan.run_id,
                config=plan.config.model_dump(mode="json"),
                n_samples=len(dataset.train),
                n_features=dataset.n_features,
            )
        )
        # Nas tarefas sem alvo, uma coluna-alvo é só referência: nunca chega ao treino.
        target = dataset.train.y if dataset.task_type.requires_target else None
        start = time.perf_counter()
        try:
            outcome = Trainer(self._bus).train(
                plan.model,
                dataset.train.X,
                target,
                run_id=plan.run_id,
                sampler=plan.sampler,
                probe=plan.grid.points if plan.grid is not None else None,
                control=control,
            )
            reports = self._evaluate(plan)
            duration = time.perf_counter() - start
            self._bus.publish(
                EvaluationCompleted(
                    run_id=plan.run_id,
                    train_report=to_json(reports[Partition.TRAIN]),
                    test_report=to_json(reports[Partition.TEST]),
                )
            )
            result = self._result(plan, outcome, reports, duration)
        except Exception as error:
            self._bus.publish(
                RunFailed(run_id=plan.run_id, message=str(error), error_type=type(error).__name__)
            )
            raise
        finally:
            plan.recorder.detach(self._bus)
            with self._lock:
                self._active.pop(plan.run_id, None)
        self._keep(plan.run_id, RunArtifacts(plan.model, dataset, plan.config, result))
        primary = result.primary
        summary: dict[str, Any] = {} if primary is None else {primary.name: primary.value}
        summary["steps"] = outcome.steps
        self._bus.publish(
            RunFinished(run_id=plan.run_id, duration_seconds=duration, summary=summary)
        )
        logger.info(
            "experiência concluída",
            extra={
                "run_id": plan.run_id,
                "model": plan.model.info.name,
                "seed": plan.config.seed,
                **summary,
            },
        )
        return result

    def _evaluate(self, plan: _Plan) -> dict[Partition, EvaluationReport]:
        dataset = plan.dataset
        evaluator = evaluator_for(dataset.task_type)
        return {
            part: evaluator.evaluate(
                plan.model,
                split.X,
                split.y,
                partition=part,
                task_type=dataset.task_type,
                class_labels=dataset.class_labels,
            )
            for part, split in dataset.splits.items()
        }

    def _result(
        self,
        plan: _Plan,
        outcome: TrainingOutcome,
        reports: dict[Partition, EvaluationReport],
        duration: float,
    ) -> ExperimentResult:
        model, dataset = plan.model, plan.dataset
        state = model.state()
        history = plan.recorder.steps()
        animation = None
        if plan.grid is not None:
            snapshots = [(record.step, record.state) for record in plan.recorder.snapshots()]
            if not snapshots:  # treinou de uma vez: só há o modelo final
                snapshots = [(0, model.snapshot_state(plan.grid.points))]
            animation = build_animation(
                dataset,
                plan.grid,
                snapshots,
                losses={record.step: record.loss for record in history},
                classes=model.classes(),
            )
        embedding = None
        if dataset.task_type is TaskType.DIMENSIONALITY_REDUCTION:
            # Na PCA, os eixos são as componentes que aprendeu; no t-SNE, não têm significado.
            axis = "Component" if "components" in state else "Dimension"
            embedding = embedding_map(dataset, model.transform(dataset.train.X), axis)
        training = list(model.notices)
        if outcome.stopped_early:
            training.insert(
                0,
                Notice.info(
                    "stopped_early",
                    f"Training stopped on request at step {format_count(outcome.steps)}: the model "
                    "kept what it had learned so far.",
                ),
            )
        return ExperimentResult(
            run_id=plan.run_id,
            config=plan.config,
            dataset_name=dataset.name,
            task_type=dataset.task_type,
            model_name=model.info.name,
            model_display_name=model.info.display_name,
            hyperparameters=model.hyperparameters.model_dump(mode="json"),
            feature_names=dataset.feature_names,
            class_labels=dataset.class_labels,
            reports=reports,
            model_state=state,
            notices=(*training, *compare_train_and_test(reports)),
            duration_seconds=duration,
            history=history,
            stopped_early=outcome.stopped_early,
            animation=animation,
            training_notices=tuple(training),
            embedding=embedding,
            learned=learned_view(
                state,
                task_type=dataset.task_type,
                feature_names=dataset.feature_names,
                class_labels=dataset.class_labels,
                target_name=dataset.target_name,
            ),
            main_partition=_main_partition(reports),
            network=network_trace(
                dataset.n_features,
                [(record.step, record.state) for record in plan.recorder.snapshots()],
            ),
        )

    def _keep(self, run_id: str, artifacts: RunArtifacts) -> None:
        with self._lock:
            self._runs[run_id] = artifacts
            self._runs.move_to_end(run_id)
            while len(self._runs) > KEPT_RUNS:
                self._runs.popitem(last=False)


def _main_partition(reports: Mapping[Partition, EvaluationReport]) -> Partition:
    """O teste, se tiver métricas; senão (um modelo que só conhece o treino), o treino."""
    test = reports.get(Partition.TEST)
    return Partition.TEST if test is not None and test.metrics else Partition.TRAIN
