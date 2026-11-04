"""Rotas do histórico de runs (ecrã Histórico): registar, guardar, comparar e reproduzir.

Cada run que termina já fica registado sozinho (ADR 0020). Aqui consulta-se o histórico,
guarda-se um run com o modelo e o treino passo a passo, e abre-se, compara-se, reproduz-se,
muda-se o nome ou apaga-se um run.
"""

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.requests import CompareRequest, RenameRequest, SaveRequest
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.core.services import (
    ExperimentResult,
    JobState,
    ReinforcementResult,
    RunComparison,
    RunEntry,
    TaskType,
    TrainingStory,
)

router = ServiceRouter(prefix="/history", tags=["history"], responses=ERROR_RESPONSES)


@router.get("/runs")
def runs(
    lab: LabDep,
    task_type: TaskType | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> tuple[RunEntry, ...]:
    """O histórico, do run mais recente para o mais antigo (filtrado, se se pedir)."""
    return lab.history.runs(task_type=task_type, dataset_name=dataset_name, model_name=model_name)


@router.get("/runs/{run_id}")
def entry(lab: LabDep, run_id: str) -> RunEntry:
    """A entrada de um run no histórico."""
    return lab.history.entry(run_id)


@router.get("/runs/{run_id}/result")
def open(lab: LabDep, run_id: str) -> ExperimentResult:  # o nome do método do serviço
    """O resultado de um run, pronto a mostrar (um run guardado volta com o modelo)."""
    return lab.history.open(run_id)


@router.get("/runs/{run_id}/story")
def story(lab: LabDep, run_id: str) -> TrainingStory:
    """O que aconteceu no treino de um run, em capítulos curtos com texto e visuais."""
    return lab.history.story(run_id)


@router.get("/runs/{run_id}/reinforcement-result")
def open_reinforcement(lab: LabDep, run_id: str) -> ReinforcementResult:
    """O resultado de um run de reinforcement learning (um run guardado volta com a política)."""
    return lab.history.open_reinforcement(run_id)


@router.post("/runs/{run_id}/save")
def save(lab: LabDep, run_id: str, request: SaveRequest) -> RunEntry:
    """Guarda o modelo treinado e o treino passo a passo de um run desta sessão."""
    return lab.history.save(run_id, request.name)


@router.patch("/runs/{run_id}")
def rename(lab: LabDep, run_id: str, request: RenameRequest) -> RunEntry:
    """Muda o nome de um run."""
    return lab.history.rename(run_id, request.name)


@router.delete("/runs/{run_id}", status_code=204)
def delete(lab: LabDep, run_id: str) -> None:
    """Apaga um run do histórico, com os seus ficheiros."""
    lab.history.delete(run_id)


@router.post("/comparison")
def compare(lab: LabDep, request: CompareRequest) -> RunComparison:
    """Os runs lado a lado: as métricas, as escolhas que mudam e as curvas da loss."""
    return lab.history.compare(request.run_ids)


@router.post("/runs/{run_id}/reproduction", status_code=202)
def reproduce(lab: LabDep, run_id: str) -> JobState:
    """Volta a treinar um run com a mesma configuração e a mesma seed, em segundo plano."""
    return lab.history.reproduce(run_id).state()
