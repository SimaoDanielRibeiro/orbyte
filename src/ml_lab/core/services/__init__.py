"""Serviços (casos de uso): a única porta de entrada do núcleo.

A interface simples chama-os diretamente; a API (fase 7) chama-os para a interface
aprimorada. Como a interface só pode importar ``ml_lab.core.services`` e
``ml_lab.core.schemas`` (regra verificada em ``tests/architecture/``), este pacote
também reexporta os tipos de que ela precisa: os resultados que os serviços
devolvem, as configurações que recebem e o erro base do ML Lab.
"""

from ml_lab.core.data.boundary import TablePreview
from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.cleaning.suggestions import CleaningSuggestion
from ml_lab.core.data.explanations import Explanation, explain
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.lineage import LineageStep, Stage
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.profiling.column_profile import ColumnProfile, Histogram, ValueCount
from ml_lab.core.data.profiling.dataset_profile import (
    ClassBalance,
    CorrelationMatrix,
    DatasetProfile,
    GroupMeans,
    MissingMap,
)
from ml_lab.core.data.projection.projection_2d import ProjectionConfig
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.task.task_detection import TaskSuggestion
from ml_lab.core.data.transforms.imputation import ImputationStrategy
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig
from ml_lab.core.data.transforms.scaling import ScalingMethod
from ml_lab.core.data.validation.checks import ValidationReport
from ml_lab.core.evaluation.report import ConfusionMatrix, EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import MLLabError
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Implementation, ModelFamily
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.services.catalog import CatalogService, ModelCard
from ml_lab.core.services.datasets import (
    ColumnInfo,
    DatasetService,
    DatasetSummary,
    GeneratorSummary,
    OperationSummary,
    PartitionSummary,
    PreparationSummary,
    ProjectionPoints,
    SampleSummary,
)
from ml_lab.core.services.experiments import ExperimentResult, ExperimentService
from ml_lab.core.services.lab import Lab

__all__ = [
    "CatalogService",
    "ClassBalance",
    "CleaningStep",
    "CleaningSuggestion",
    "ColumnInfo",
    "ColumnKind",
    "ColumnProfile",
    "ConfusionMatrix",
    "CorrelationMatrix",
    "DatasetProfile",
    "DatasetService",
    "DatasetSummary",
    "EvaluationReport",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentService",
    "Explanation",
    "GeneratorSummary",
    "GroupMeans",
    "Histogram",
    "Implementation",
    "ImputationStrategy",
    "Lab",
    "LineageStep",
    "MLLabError",
    "MetricResult",
    "MissingMap",
    "ModelCard",
    "ModelFamily",
    "ModelSpec",
    "Notice",
    "NoticeLevel",
    "OperationSummary",
    "Partition",
    "PartitionSummary",
    "PreparationConfig",
    "PreparationSummary",
    "ProjectionConfig",
    "ProjectionPoints",
    "SampleSummary",
    "ScalingMethod",
    "SourceSpec",
    "SplitConfig",
    "SplitStrategy",
    "Stage",
    "TablePreview",
    "TaskSuggestion",
    "TaskType",
    "TransformConfig",
    "ValidationReport",
    "ValueCount",
    "explain",
]
