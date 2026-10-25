"""Demonstrações dos modelos: cada modelo a trabalhar num conjunto de exemplo pequeno.

A página de cada modelo mostra como ele funciona com um exemplo a sério: treina-o num
conjunto sintético pequeno da sua tarefa (as luas, os círculos, as nuvens, uma onda…) e
devolve o mesmo que um run devolve, o campo em cada fotografia do treino, a loss, o que
aprendeu, a avaliação e os gráficos. Mudar um hiperparâmetro e voltar a pedir mostra o efeito
dele nos mesmos pontos (ADR 0028).

Uma demonstração não é um run. Treina num serviço de experiências só seu, com um bus de
eventos sem ninguém a ouvir: nada vai para o histórico, e os dados não aparecem no ecrã Dados.
As últimas ficam guardadas, para voltar a um valor já visto ser imediato.

Os conjuntos de exemplo são uma decisão didática, por isso estão aqui (no backend) e não na
interface: cada tarefa tem os seus, e o primeiro é o que mostra melhor a ideia da tarefa.
"""

import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ml_lab.config import Settings
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.io.synthetic_source import TARGET
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.evaluation.report import EvaluationReport
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import LabModel
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import ExperimentService
from ml_lab.core.training.recorder import StepRecord
from ml_lab.core.visualization_data.animation import TrainingAnimation
from ml_lab.core.visualization_data.embedding import Embedding
from ml_lab.core.visualization_data.evaluation import EvaluationCharts, evaluation_charts
from ml_lab.core.visualization_data.learned import LearnedView
from ml_lab.core.visualization_data.network import NetworkTrace

KEPT_DEMOS: Final = 32
"""Quantas demonstrações ficam guardadas (as mais antigas saem primeiro)."""

TASK_PREFERENCE: Final = (
    TaskType.BINARY_CLASSIFICATION,
    TaskType.REGRESSION,
    TaskType.CLUSTERING,
    TaskType.ANOMALY_DETECTION,
    TaskType.DIMENSIONALITY_REDUCTION,
    TaskType.MULTICLASS_CLASSIFICATION,
)
"""A tarefa que a demonstração mostra primeiro, num modelo que resolve várias: a binária, a
mais fácil de ver no plano (duas cores, uma fronteira)."""


@dataclass(frozen=True, slots=True)
class DemoExample:
    """Um conjunto de exemplo para ver um modelo a trabalhar.

    Attributes:
        name: O identificador (o que se pede em ``example``).
        title: O nome a mostrar.
        description: O que o conjunto tem de interessante, numa frase.
        task_type: A tarefa que o modelo resolve nele.
    """

    name: str
    title: str
    description: str
    task_type: TaskType


@dataclass(frozen=True, slots=True)
class _Example:
    """Um conjunto de exemplo e como se gera: o gerador, os parâmetros e a seed."""

    info: DemoExample
    generator: str
    parameters: Mapping[str, float]
    seed: int

    def config(self, model: str, hyperparameters: dict[str, Any]) -> ExperimentConfig:
        """A experiência da demonstração: estes dados, o alvo como alvo (ou referência)."""
        source = SourceSpec(
            "synthetic",
            {"generator": self.generator, "seed": self.seed, "parameters": dict(self.parameters)},
        )
        return ExperimentConfig(
            dataset=source,
            preparation=PreparationConfig(target=TARGET, task_type=self.info.task_type),
            model=ModelSpec(name=model, hyperparameters=hyperparameters),
            seed=self.seed,
        )


EXAMPLES: Final = (
    _Example(
        DemoExample(
            "moons",
            "Luas",
            "Duas meias-luas encaixadas: nenhuma reta as separa.",
            TaskType.BINARY_CLASSIFICATION,
        ),
        "moons",
        {"n_samples": 200, "noise": 0.25},
        3,
    ),
    _Example(
        DemoExample(
            "circles",
            "Círculos",
            "Um círculo dentro de outro: a fronteira tem de dar a volta.",
            TaskType.BINARY_CLASSIFICATION,
        ),
        "circles",
        {"n_samples": 200, "noise": 0.08, "factor": 0.5},
        3,
    ),
    _Example(
        DemoExample(
            "blobs",
            "Três nuvens",
            "Três grupos de pontos, um por classe, que se tocam nas pontas.",
            TaskType.MULTICLASS_CLASSIFICATION,
        ),
        "blobs",
        {"n_samples": 240, "centers": 3, "cluster_std": 1.8, "n_features": 2},
        5,
    ),
    _Example(
        DemoExample(
            "wave",
            "Onda",
            "Um alvo que sobe e desce com a feature: uma reta não o consegue seguir.",
            TaskType.REGRESSION,
        ),
        "wave",
        {"n_samples": 150, "noise": 0.25},
        3,
    ),
    _Example(
        DemoExample(
            "line",
            "Reta com ruído",
            "Um alvo que cresce a direito com a feature, mais ruído: o caso do modelo linear.",
            TaskType.REGRESSION,
        ),
        "regression",
        {"n_samples": 150, "n_features": 1, "noise": 1.5},
        3,
    ),
    _Example(
        DemoExample(
            "clusters",
            "Três grupos",
            "Três nuvens sem rótulo: o modelo tem de as encontrar sozinho.",
            TaskType.CLUSTERING,
        ),
        "blobs",
        {"n_samples": 240, "centers": 3, "cluster_std": 1.2, "n_features": 2},
        5,
    ),
    _Example(
        DemoExample(
            "moon_clusters",
            "Luas sem rótulo",
            "Dois grupos com forma de lua: quem corta a direito separa-os mal.",
            TaskType.CLUSTERING,
        ),
        "moons",
        {"n_samples": 240, "noise": 0.06},
        3,
    ),
    _Example(
        DemoExample(
            "outliers",
            "Anomalias",
            "Duas nuvens normais e alguns pontos espalhados à volta delas.",
            TaskType.ANOMALY_DETECTION,
        ),
        "anomalies",
        {"n_samples": 240, "contamination": 0.06, "n_features": 2},
        4,
    ),
    _Example(
        DemoExample(
            "high_dimensional",
            "Nuvens em 5 dimensões",
            "Quatro grupos num espaço de 5 features: a projeção tem de os manter separados em 2.",
            TaskType.DIMENSIONALITY_REDUCTION,
        ),
        "blobs",
        {"n_samples": 240, "centers": 4, "cluster_std": 1.5, "n_features": 5},
        5,
    ),
)
"""Os conjuntos de exemplo, pela ordem em que aparecem (o primeiro de cada tarefa é o usado
por omissão)."""


@dataclass(frozen=True, slots=True)
class ModelDemo:
    """Um modelo a trabalhar num conjunto de exemplo (o que a página do modelo mostra).

    Attributes:
        model_name: O modelo, no registo.
        model_display_name: O nome a mostrar.
        example: O conjunto de exemplo usado.
        task_type: A tarefa (a do conjunto de exemplo).
        hyperparameters: Os hiperparâmetros usados, completos.
        feature_names: As features que o modelo recebeu.
        class_labels: Na classificação, as classes (no clustering e nas anomalias, as da
            coluna de referência).
        animation: O campo (ou a curva) em cada fotografia do treino; ``None`` na redução de
            dimensionalidade, que mostra o mapa.
        history: Cada passo do treino, com a loss (vazio nos modelos que treinam de uma vez).
        learned: O que o modelo aprendeu: pesos, árvore, centros…
        network: Numa rede neuronal, o que cada camada fez em cada fotografia.
        embedding: Na redução de dimensionalidade, o mapa dos pontos.
        reports: A avaliação no treino e no teste.
        main_partition: A parte cujo relatório resume a demonstração.
        charts: Os gráficos da avaliação (ROC, resíduos, silhueta…).
        notices: Os avisos do treino e da comparação entre o treino e o teste.
        duration_seconds: Quanto demorou a treinar e a avaliar.
    """

    model_name: str
    model_display_name: str
    example: DemoExample
    task_type: TaskType
    hyperparameters: dict[str, Any]
    feature_names: tuple[str, ...]
    class_labels: tuple[str, ...]
    animation: TrainingAnimation | None
    history: tuple[StepRecord, ...]
    learned: LearnedView
    network: NetworkTrace | None
    embedding: Embedding | None
    reports: Mapping[Partition, EvaluationReport]
    main_partition: Partition
    charts: EvaluationCharts
    notices: tuple[Notice, ...]
    duration_seconds: float


def examples_for(model: type[LabModel[Any]]) -> tuple[DemoExample, ...]:
    """Os conjuntos de exemplo de ``model``: o de omissão primeiro, depois os outros."""
    return tuple(example.info for example in _examples(model))


def _examples(model: type[LabModel[Any]]) -> list[_Example]:
    tasks = model.info.task_types
    main = next(task for task in TASK_PREFERENCE if task in tasks)
    usable = [example for example in EXAMPLES if example.info.task_type in tasks]
    # A tarefa principal primeiro; dentro de cada tarefa, a ordem de EXAMPLES.
    return sorted(usable, key=lambda example: example.info.task_type is not main)


class ModelDemos:
    """Treina as demonstrações, longe do histórico, e guarda as últimas.

    Args:
        settings: As definições (as demonstrações têm um serviço de datasets só seu).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # Um bus novo, sem ninguém a ouvir: nenhum registo de experiências vê estes treinos.
        self._experiments = ExperimentService(DatasetService(settings))
        self._kept: OrderedDict[str, ModelDemo] = OrderedDict()
        self._lock = threading.Lock()

    def run(
        self, model: type[LabModel[Any]], hyperparameters: Mapping[str, Any], example: str | None
    ) -> ModelDemo:
        """A demonstração de ``model`` com estes hiperparâmetros, no conjunto ``example``.

        Raises:
            HyperparameterError: Se os hiperparâmetros não forem válidos.
            InvalidParameterError: Se o conjunto de exemplo não servir para o modelo.
        """
        info = model.info
        chosen = self._choose(model, example)
        values = info.hyperparameters.parse(dict(hyperparameters), model=info.name)
        complete = values.model_dump(mode="json")
        key = json.dumps([info.name, chosen.info.name, complete], sort_keys=True)
        # Um treino de cada vez: dois pedidos iguais seguidos treinam uma só vez.
        with self._lock:
            kept = self._kept.get(key)
            if kept is None:
                kept = self._train(chosen, info.name, complete)
                self._kept[key] = kept
                while len(self._kept) > KEPT_DEMOS:
                    self._kept.popitem(last=False)
            self._kept.move_to_end(key)
            return kept

    def _choose(self, model: type[LabModel[Any]], example: str | None) -> _Example:
        usable = _examples(model)
        if example is None:
            return usable[0]
        for candidate in usable:
            if candidate.info.name == example:
                return candidate
        names = ", ".join(candidate.info.name for candidate in usable)
        raise InvalidParameterError(
            f"'{model.info.display_name}' não tem o exemplo '{example}'. Escolhe um destes: "
            f"{names}."
        )

    def _train(self, example: _Example, model: str, hyperparameters: dict[str, Any]) -> ModelDemo:
        result = self._experiments.run(example.config(model, hyperparameters))
        artifacts = self._experiments.artifacts(result.run_id)
        return ModelDemo(
            model_name=result.model_name,
            model_display_name=result.model_display_name,
            example=example.info,
            task_type=result.task_type,
            hyperparameters=result.hyperparameters,
            feature_names=result.feature_names,
            class_labels=result.class_labels,
            animation=result.animation,
            history=result.history,
            learned=result.learned,
            network=result.network,
            embedding=result.embedding,
            reports=result.reports,
            main_partition=result.main_partition,
            charts=evaluation_charts(artifacts.model, artifacts.dataset),
            notices=result.notices,
            duration_seconds=result.duration_seconds,
        )
