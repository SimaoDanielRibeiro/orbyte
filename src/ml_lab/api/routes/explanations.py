"""Rotas dos textos da interface: escrevem-se uma vez, no backend.

São dois pedidos que o frontend faz uma vez, ao arrancar, e guarda:

- ``GET /vocabulary``: o nome a mostrar de cada valor de cada enum (``TaskType``,
  ``ModelFamily``…) e o que muda com cada tarefa (a métrica principal, o modelo sugerido);
- ``GET /explanations``: todos os temas do painel "Porquê?".

``GET /explanations/{topic}`` dá um tema só.
"""

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import ServiceRouter
from ml_lab.core.services import Explanation, Vocabulary
from ml_lab.core.services import explain as explain_topic
from ml_lab.core.services import topics as all_topics
from ml_lab.core.services import vocabulary as interface_vocabulary

router = ServiceRouter(tags=["explanations"], responses=ERROR_RESPONSES)


@router.get("/vocabulary")
def vocabulary() -> Vocabulary:
    """O nome a mostrar de cada valor de cada enum, e o que muda com cada tarefa."""
    return interface_vocabulary()


@router.get("/explanations")
def topics() -> dict[str, Explanation]:
    """Todos os temas do painel "Porquê?", pelo nome."""
    return all_topics()


@router.get("/explanations/{topic}")
def explain(topic: str) -> Explanation:
    """A explicação didática de um tema (o título e o texto)."""
    return explain_topic(topic)
