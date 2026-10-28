"""Serviços (casos de uso): a única porta de entrada do núcleo.

A API chama-os, e a interface web fala com a API. Como a API só pode importar
``ml_lab.core.services`` e ``ml_lab.core.schemas`` (regra verificada em
``tests/architecture/``), este pacote também reexporta os tipos de que ela precisa: os
resultados que os serviços devolvem, as configurações que recebem e o erro base do ML Lab.
"""

from ml_lab.core.automl.data_agent import DataDecision
from ml_lab.core.automl.model_agent import SkippedModel
from ml_lab.core.automl.search_log import (
    AgentMessage,
    AgentRole,
    Candidate,
    MessageKind,
    Participant,
    ParticipantKind,
    SearchPhase,
    WorkInProgress,
)
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
from ml_lab.core.evaluation.explanations import FormulaSymbol
from ml_lab.core.evaluation.metrics.base import MetricTerm
from ml_lab.core.evaluation.report import ConfusionMatrix, EvaluationReport, MetricResult
from ml_lab.core.evaluation.validation import (
    CrossValidation,
    CurvePoint,
    FoldScore,
    LearningCurve,
)
from ml_lab.core.explainability.partial_dependence import PartialDependence
from ml_lab.core.explainability.permutation import FeatureImportance, PermutationImportance
from ml_lab.core.foundation.errors import (
    DataError,
    MLLabError,
    TrackingError,
    TrainingStateError,
)
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Implementation, ModelFamily
from ml_lab.core.rl.environments.base import CellKind, GridLayout
from ml_lab.core.schemas.automl import AutoMLRequest, SearchSettings
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.schemas.reinforcement import AgentSpec, EnvironmentSpec, ReinforcementConfig
from ml_lab.core.services.analysis import AnalysisService, Contribution, RowExplanation
from ml_lab.core.services.automl import (
    AgentModel,
    AutoMLOptions,
    AutoMLPlan,
    AutoMLResult,
    AutoMLService,
    AutoMLUpdate,
    ModelAgentCard,
    ObjectiveOption,
)
from ml_lab.core.services.catalog import CatalogService, DocSection, MetricCard, ModelCard
from ml_lab.core.services.comparison import (
    ConfigDifference,
    LossSeries,
    MetricComparison,
    RunComparison,
)
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
from ml_lab.core.services.demos import DemoExample, ModelDemo
from ml_lab.core.services.experiments import ExperimentResult, ExperimentService, TrainingUpdate
from ml_lab.core.services.explanations import explain, topics
from ml_lab.core.services.history import HistoryService
from ml_lab.core.services.lab import Lab
from ml_lab.core.services.laboratory import (
    DemoCard,
    DemoControl,
    DemoVariant,
    KnnDemo,
    LaboratoryDemo,
    LaboratoryService,
    LearningRateDemo,
    LossCurve,
    OverfittingDemo,
    Points,
    RegularizationDemo,
)
from ml_lab.core.services.reinforcement import (
    AgentCard,
    EnvironmentCard,
    ReinforcementResult,
    ReinforcementService,
    ReinforcementUpdate,
)
from ml_lab.core.services.vocabulary import TaskInfo, Vocabulary, vocabulary
from ml_lab.core.tracking.store import RunEntry
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
from ml_lab.core.visualization_data.learned import (
    LearnedView,
    LinearWeights,
    Matrix,
    Property,
    TreeNode,
)
from ml_lab.core.visualization_data.network import (
    LayerFrame,
    LayerInfo,
    NetworkFrame,
    NetworkTrace,
)
from ml_lab.core.visualization_data.policy import PolicyFrame, PolicyTrace, RewardCurve

__all__ = [
    "AgentCard",
    "AgentMessage",
    "AgentModel",
    "AgentRole",
    "AgentSpec",
    "AnalysisService",
    "AutoMLOptions",
    "AutoMLPlan",
    "AutoMLRequest",
    "AutoMLResult",
    "AutoMLService",
    "AutoMLUpdate",
    "Candidate",
    "CatalogService",
    "CellKind",
    "ClassBalance",
    "CleaningStep",
    "CleaningSuggestion",
    "ColumnInfo",
    "ColumnKind",
    "ColumnProfile",
    "ConfigDifference",
    "ConfusionMatrix",
    "Contribution",
    "CorrelationMatrix",
    "CrossValidation",
    "Curve",
    "CurvePoint",
    "DataDecision",
    "DataError",
    "DatasetProfile",
    "DatasetService",
    "DatasetSummary",
    "DemoCard",
    "DemoControl",
    "DemoExample",
    "DemoVariant",
    "Dendrogram",
    "Distribution",
    "DocSection",
    "Embedding",
    "EnvironmentCard",
    "EnvironmentSpec",
    "EvaluationCharts",
    "EvaluationReport",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentService",
    "Explanation",
    "FeatureImportance",
    "FieldFrame",
    "FoldScore",
    "FormulaSymbol",
    "GeneratorSummary",
    "GridLayout",
    "GroupMeans",
    "Histogram",
    "HistoryService",
    "Implementation",
    "ImputationStrategy",
    "JobState",
    "JobStatus",
    "KnnDemo",
    "Lab",
    "LaboratoryDemo",
    "LaboratoryService",
    "LayerFrame",
    "LayerInfo",
    "LearnedView",
    "LearningCurve",
    "LearningRateDemo",
    "LineageStep",
    "LinearWeights",
    "LossCurve",
    "LossSeries",
    "MLLabError",
    "Matrix",
    "MessageKind",
    "MetricCard",
    "MetricComparison",
    "MetricResult",
    "MetricTerm",
    "MissingMap",
    "ModelAgentCard",
    "ModelCard",
    "ModelDemo",
    "ModelFamily",
    "ModelSpec",
    "NetworkFrame",
    "NetworkTrace",
    "Notice",
    "NoticeLevel",
    "ObjectiveOption",
    "OperationSummary",
    "Outline",
    "OverfittingDemo",
    "PartialDependence",
    "Participant",
    "ParticipantKind",
    "Partition",
    "PartitionSummary",
    "PermutationImportance",
    "PointCloud",
    "Points",
    "PolicyFrame",
    "PolicyTrace",
    "PreparationConfig",
    "PreparationSummary",
    "Progress",
    "ProjectionConfig",
    "ProjectionPoints",
    "Property",
    "RegularizationDemo",
    "ReinforcementConfig",
    "ReinforcementResult",
    "ReinforcementService",
    "ReinforcementUpdate",
    "Residuals",
    "RewardCurve",
    "RowExplanation",
    "RunComparison",
    "RunEntry",
    "SampleSummary",
    "ScalingMethod",
    "SearchPhase",
    "SearchSettings",
    "Silhouette",
    "SilhouetteGroup",
    "SkippedModel",
    "SourceSpec",
    "SplitConfig",
    "SplitStrategy",
    "Stage",
    "StepRecord",
    "TablePreview",
    "TaskInfo",
    "TaskOptions",
    "TaskSuggestion",
    "TaskType",
    "TrackingError",
    "TrainingAnimation",
    "TrainingJob",
    "TrainingStateError",
    "TrainingUpdate",
    "TransformConfig",
    "TreeNode",
    "ValidationReport",
    "ValueCount",
    "Vocabulary",
    "WorkInProgress",
    "explain",
    "topics",
    "vocabulary",
]
