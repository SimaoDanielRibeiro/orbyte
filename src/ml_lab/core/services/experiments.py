"""Serviço de experiências: preparar os dados, treinar, avaliar e anunciar o run.

É o fluxo completo da secção "Fluxo de uma experiência" do plano, numa só chamada:

1. resolve a seed raiz e as seeds de cada componente (ADR 0006);
2. prepara os dados com a configuração (a camada de dados devolve o ``Dataset``);
3. vai buscar o modelo ao registo e valida os hiperparâmetros;
4. anuncia o início do run, treina e avalia cada parte dos dados;
5. compara treino e teste (o alerta de overfitting) e anuncia o fim.

Os eventos (``RunStarted``, ``EvaluationCompleted``, ``RunFinished``, ``RunFailed``)
vão para o ``EventBus`` do serviço: quem quiser acompanhar os runs subscreve-o. Na
fase 3, o treino passo a passo publica também os passos e os snapshots.
"""

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ml_lab.core.evaluation.base import PRIMARY_METRIC, evaluator_for
from ml_lab.core.evaluation.diagnostics import compare_train_and_test
from ml_lab.core.evaluation.report import EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import ModelError
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
from ml_lab.core.models.base import MODELS
from ml_lab.core.schemas.experiment import ExperimentConfig
from ml_lab.core.services.datasets import DatasetService
from ml_lab.logging import get_logger

logger = get_logger(__name__)

MODEL_SEED: Final = "model"
"""Nome do componente do modelo: a sua seed é ``seeds.seed_for("model")``."""


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """O resultado de uma experiência, pronto a mostrar no ecrã Avaliar.

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
        notices: Avisos do treino e dos diagnósticos.
        duration_seconds: Quanto tempo demorou o treino e a avaliação.
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

    @property
    def seed(self) -> int:
        """A seed raiz usada."""
        assert self.config.seed is not None  # o serviço resolve-a sempre antes de criar o resultado
        return self.config.seed

    @property
    def primary(self) -> MetricResult | None:
        """A métrica principal no teste (por exemplo, a exatidão)."""
        name = PRIMARY_METRIC.get(self.task_type)
        test = self.reports.get(Partition.TEST)
        return None if name is None or test is None else test.metric(name)


class ExperimentService:
    """Corre experiências.

    Args:
        datasets: O serviço de datasets (prepara os dados e guarda-os em memória).
        bus: Para onde publicar os eventos dos runs; por omissão, um novo.
    """

    def __init__(self, datasets: DatasetService, *, bus: EventBus | None = None) -> None:
        self._datasets = datasets
        self._bus = bus or EventBus()

    @property
    def bus(self) -> EventBus:
        """O ``EventBus`` onde os runs anunciam o que acontece."""
        return self._bus

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Corre a experiência descrita por ``config``.

        Raises:
            ComponentNotFoundError: Se o modelo não existir.
            HyperparameterError: Se os hiperparâmetros não forem válidos.
            ModelError: Se o modelo não resolver o tipo de tarefa dos dados.
            DataError: Se os dados não servirem para a preparação pedida.
        """
        seeds = SeedManager.create(config.seed)
        config = config.model_copy(update={"seed": seeds.seed})
        model_class = MODELS.get(config.model.name)
        info = model_class.info
        hyperparameters = info.hyperparameters.parse(config.model.hyperparameters, model=info.name)
        dataset = self._datasets.prepared(config.dataset, config.preparation, seeds.seed).dataset
        if dataset.task_type not in info.task_types:
            raise ModelError(
                f"'{info.display_name}' não resolve {dataset.task_type.label.lower()}. Escolhe um "
                "modelo do catálogo filtrado por esta tarefa."
            )

        run_id = f"run-{uuid.uuid4().hex[:8]}"
        self._bus.publish(
            RunStarted(
                run_id=run_id,
                config=config.model_dump(mode="json"),
                n_samples=len(dataset.train),
                n_features=dataset.n_features,
            )
        )
        start = time.perf_counter()
        try:
            model = model_class(hyperparameters, seed=seeds.seed_for(MODEL_SEED))
            model.fit(dataset.train.X, dataset.train.y)
            evaluator = evaluator_for(dataset.task_type)
            reports = {
                part: evaluator.evaluate(
                    model,
                    split.X,
                    split.y,
                    partition=part,
                    task_type=dataset.task_type,
                    class_labels=dataset.class_labels,
                )
                for part, split in dataset.splits.items()
            }
        except Exception as error:
            self._bus.publish(
                RunFailed(run_id=run_id, message=str(error), error_type=type(error).__name__)
            )
            raise
        duration = time.perf_counter() - start

        self._bus.publish(
            EvaluationCompleted(
                run_id=run_id,
                train_report=reports[Partition.TRAIN].to_dict(),
                test_report=reports[Partition.TEST].to_dict(),
            )
        )
        result = ExperimentResult(
            run_id=run_id,
            config=config,
            dataset_name=dataset.name,
            task_type=dataset.task_type,
            model_name=info.name,
            model_display_name=info.display_name,
            hyperparameters=hyperparameters.model_dump(mode="json"),
            feature_names=dataset.feature_names,
            class_labels=dataset.class_labels,
            reports=reports,
            model_state=model.state(),
            notices=(*model.notices, *compare_train_and_test(reports)),
            duration_seconds=duration,
        )
        primary = result.primary
        summary = {} if primary is None else {primary.name: primary.value}
        self._bus.publish(RunFinished(run_id=run_id, duration_seconds=duration, summary=summary))
        logger.info(
            "experiência concluída",
            extra={"run_id": run_id, "model": info.name, "seed": seeds.seed, **summary},
        )
        return result
