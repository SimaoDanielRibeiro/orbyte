"""Rotas dos dados (ecrãs Dados, Explorar e Preparar): o ``DatasetService`` por HTTP."""

from typing import Annotated

from fastapi import Query, UploadFile

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.requests import GenerateRequest, PrepareRequest
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.core.services import (
    CleaningSuggestion,
    DatasetProfile,
    DatasetSummary,
    GeneratorSummary,
    OperationSummary,
    PreparationSummary,
    SampleSummary,
    SourceSpec,
    TablePreview,
    TaskOptions,
    TaskSuggestion,
    TaskType,
)

router = ServiceRouter(prefix="/datasets", tags=["datasets"], responses=ERROR_RESPONSES)

Target = Annotated[str | None, Query(description="A coluna-alvo (vazio: sem alvo).")]


@router.get("/samples")
def samples(lab: LabDep) -> list[SampleSummary]:
    """Os datasets de exemplo."""
    return lab.datasets.samples()


@router.get("/generators")
def generators(lab: LabDep) -> list[GeneratorSummary]:
    """Os geradores de dados sintéticos, com os parâmetros de cada um (em JSON Schema)."""
    return lab.datasets.generators()


@router.get("/cleaning-operations")
def cleaning_operations(lab: LabDep) -> list[OperationSummary]:
    """As operações de limpeza disponíveis."""
    return lab.datasets.cleaning_operations()


@router.get("/task-options")
def task_options(lab: LabDep, task_type: TaskType, target: Target = None) -> TaskOptions:
    """O que muda com a tarefa escolhida: a coluna de referência e as divisões possíveis."""
    return lab.datasets.task_options(task_type, target)


@router.get("")
def loaded(lab: LabDep) -> list[DatasetSummary]:
    """Os datasets carregados, do mais recente para o mais antigo."""
    return lab.datasets.loaded()


@router.post("")
def load(lab: LabDep, spec: SourceSpec) -> DatasetSummary:
    """Carrega os dados de uma origem (um exemplo, um ficheiro já enviado, um gerador)."""
    return lab.datasets.load(spec)


@router.post("/samples/{name}")
def load_sample(lab: LabDep, name: str) -> DatasetSummary:
    """Carrega um dataset de exemplo."""
    return lab.datasets.load_sample(name)


@router.post("/generated")
def generate(lab: LabDep, request: GenerateRequest) -> DatasetSummary:
    """Gera um dataset sintético."""
    return lab.datasets.generate(request.generator, request.parameters, request.seed)


@router.post("/uploads")
def upload(lab: LabDep, file: UploadFile) -> DatasetSummary:
    """Envia um ficheiro CSV ou Excel e carrega-o.

    O ficheiro fica guardado na pasta dos uploads, com o início do seu SHA-256 no nome:
    enviar duas vezes o mesmo ficheiro não o duplica.
    """
    return lab.datasets.upload(file.filename or "dados.csv", file.file.read())


@router.get("/{dataset_id}")
def summary(lab: LabDep, dataset_id: str) -> DatasetSummary:
    """O resumo de um dataset carregado."""
    return lab.datasets.summary(dataset_id)


@router.get("/{dataset_id}/preview")
def preview(
    lab: LabDep, dataset_id: str, n_rows: Annotated[int, Query(ge=1, le=500)] = 20
) -> TablePreview:
    """As primeiras linhas, tal como foram lidas."""
    return lab.datasets.preview(dataset_id, n_rows)


@router.get("/{dataset_id}/profile")
def profile(lab: LabDep, dataset_id: str, target: Target = None) -> DatasetProfile:
    """O perfil do dataset (com o equilíbrio das classes, se houver alvo)."""
    return lab.datasets.profile(dataset_id, target)


@router.get("/{dataset_id}/task-suggestion")
def suggest_task(lab: LabDep, dataset_id: str, target: Target = None) -> TaskSuggestion:
    """O tipo de tarefa sugerido para o alvo, e as alternativas permitidas."""
    return lab.datasets.suggest_task(dataset_id, target)


@router.get("/{dataset_id}/cleaning-suggestion")
def suggest_cleaning(
    lab: LabDep, dataset_id: str, target: Target = None
) -> tuple[CleaningSuggestion, ...]:
    """Uma receita de limpeza sugerida a partir do perfil."""
    return lab.datasets.suggest_cleaning(dataset_id, target)


@router.post("/{dataset_id}/preparation")
def prepare(lab: LabDep, dataset_id: str, request: PrepareRequest) -> PreparationSummary:
    """Prepara o dataset: limpeza, divisão, transformações e projeção 2D."""
    return lab.datasets.prepare(dataset_id, request.config, request.seed)
