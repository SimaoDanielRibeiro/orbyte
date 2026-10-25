"""Rotas do catálogo de modelos (ecrã Modelo): o ``CatalogService`` por HTTP."""

from typing import Any

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.core.schemas.catalog import DemoRequest
from ml_lab.core.services import MetricCard, ModelCard, ModelDemo, TaskType

router = ServiceRouter(prefix="/catalog", tags=["catalog"], responses=ERROR_RESPONSES)


@router.get("/models")
def models(lab: LabDep, task_type: TaskType | None = None) -> list[ModelCard]:
    """Os modelos (só os que resolvem ``task_type``, se for dado), por família e nome."""
    return lab.catalog.models(task_type)


@router.get("/models/{name}")
def model(lab: LabDep, name: str) -> ModelCard:
    """O cartão de um modelo: a ficha técnica, os hiperparâmetros e a ficha didática."""
    return lab.catalog.model(name)


@router.get("/suggested-model")
def suggested_model(lab: LabDep, task_type: TaskType) -> str | None:
    """O primeiro modelo a experimentar numa tarefa: a referência que os outros têm de bater."""
    return lab.catalog.suggested_model(task_type)


@router.post("/models/{name}/hyperparameters")
def validate_hyperparameters(lab: LabDep, name: str, values: dict[str, Any]) -> dict[str, Any]:
    """Valida os hiperparâmetros e devolve-os completos (com os valores por omissão)."""
    return lab.catalog.validate_hyperparameters(name, values)


@router.post("/models/{name}/demo")
def demo(lab: LabDep, name: str, body: DemoRequest) -> ModelDemo:
    """O modelo a trabalhar num conjunto de exemplo pequeno, fora do histórico."""
    return lab.catalog.demo(name, body.hyperparameters, body.example)


@router.get("/metrics")
def metrics(lab: LabDep, task_type: TaskType | None = None) -> list[MetricCard]:
    """As métricas (só as de ``task_type``, se for dada), com a fórmula e a explicação."""
    return lab.catalog.metrics(task_type)
