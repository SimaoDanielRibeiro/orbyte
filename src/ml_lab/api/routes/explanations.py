"""Rota dos textos do painel "Porquê?": escrevem-se uma vez, no backend, para as duas interfaces."""

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import ServiceRouter
from ml_lab.core.services import Explanation
from ml_lab.core.services import explain as explain_topic

router = ServiceRouter(prefix="/explanations", tags=["explanations"], responses=ERROR_RESPONSES)


@router.get("/{topic}")
def explain(topic: str) -> Explanation:
    """A explicação didática de um tema (o título e o texto)."""
    return explain_topic(topic)
