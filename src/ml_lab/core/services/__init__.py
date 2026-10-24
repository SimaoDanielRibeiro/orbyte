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
from ml_lab.core.data.explanations import Explanation
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.lineage import LineageStep, Stage
from ml_lab.core.data.preparation import PreparationConfig, TaskOptions
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
from ml_lab.core.evaluation.validation import (
    CrossValidation,
    CurvePoint,
    FoldScore,
    LearningCurve,
)
from ml_lab.core.explainability.partial_dependence import PartialDependence
from ml_lab.core.explainability.permutation import FeatureImportance, PermutationImportance
from ml_lab.core.foundation.errors import MLLabError
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Implementation, ModelFamily
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.services.analysis import AnalysisService, Contribution, RowExplanation
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
from ml_lab.core.services.explanations import explain
from ml_lab.core.services.lab import Lab
from ml_lab.core.services.laboratory import (
    KnnDemo,
    LaboratoryService,
    LearningRateDemo,
    LossCurve,
    OverfittingDemo,
    Points,
    RegularizationDemo,
)
from ml_lab.core.training.jobs import JobState, JobStatus, TrainingJob
from ml_lab.core.training.recorder import Progress, StepRecord
from ml_lab.core.visualization_data.animation import (
    FieldFrame,
    Outline,
    PointCloud,
    TrainingAnimation,
)
from ml_lab.core.visualization_data.embedding import Embedding
from ml_lab.core.visualization_data.evaluation import (
    Curve,
    Dendrogram,
    Distribution,
    EvaluationCharts,
    Residuals,
    Silhouette,
    SilhouetteGroup,
)
from ml_lab.core.visualization_data.learned import LearnedView, LinearWeights, Matrix, Property

__all__ = [
    "AnalysisService",
    "CatalogService",
    "ClassBalance",
    "CleaningStep",
    "CleaningSuggestion",
    "ColumnInfo",
    "ColumnKind",
    "ColumnProfile",
    "ConfusionMatrix",
    "Contribution",
    "CorrelationMatrix",
    "CrossValidation",
    "Curve",
    "CurvePoint",
    "DatasetProfile",
    "DatasetService",
    "DatasetSummary",
    "Dendrogram",
    "Distribution",
    "Embedding",
    "EvaluationCharts",
    "EvaluationReport",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentService",
    "Explanation",
    "FeatureImportance",
    "FieldFrame",
    "FoldScore",
    "GeneratorSummary",
    "GroupMeans",
    "Histogram",
    "Implementation",
    "ImputationStrategy",
    "JobState",
    "JobStatus",
    "KnnDemo",
    "Lab",
    "LaboratoryService",
    "LearnedView",
    "LearningCurve",
    "LearningRateDemo",
    "LineageStep",
    "LinearWeights",
    "LossCurve",
    "MLLabError",
    "Matrix",
    "MetricResult",
    "MissingMap",
    "ModelCard",
    "ModelFamily",
    "ModelSpec",
    "Notice",
    "NoticeLevel",
    "OperationSummary",
    "Outline",
    "OverfittingDemo",
    "PartialDependence",
    "Partition",
    "PartitionSummary",
    "PermutationImportance",
    "PointCloud",
    "Points",
    "PreparationConfig",
    "PreparationSummary",
    "Progress",
    "ProjectionConfig",
    "ProjectionPoints",
    "Property",
    "RegularizationDemo",
    "Residuals",
    "RowExplanation",
    "SampleSummary",
    "ScalingMethod",
    "Silhouette",
    "SilhouetteGroup",
    "SourceSpec",
    "SplitConfig",
    "SplitStrategy",
    "Stage",
    "StepRecord",
    "TablePreview",
    "TaskOptions",
    "TaskSuggestion",
    "TaskType",
    "TrainingAnimation",
    "TrainingJob",
    "TransformConfig",
    "ValidationReport",
    "ValueCount",
    "explain",
]
