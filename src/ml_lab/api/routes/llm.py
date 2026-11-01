"""Rotas dos agentes de IA: o ``LanguageModelService`` por HTTP (fase 13, ADR 0042)."""

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.core.services import LanguageModelStatus

router = ServiceRouter(prefix="/llm", tags=["llm"], responses=ERROR_RESPONSES)


@router.get("/status")
def status(lab: LabDep) -> LanguageModelStatus:
    """Se os agentes de IA podem trabalhar: o Ollama, o modelo e o que falta fazer."""
    return lab.llm.status()
