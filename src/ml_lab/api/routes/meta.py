"""Rota sobre a própria API: para o frontend saber se ela está a correr, e em que versão."""

from dataclasses import dataclass

from ml_lab import __version__
from ml_lab.api.routing import ServiceRouter

router = ServiceRouter(tags=["meta"])


@dataclass(frozen=True, slots=True)
class Health:
    """O estado da API.

    Attributes:
        status: ``"ok"`` quando a API responde.
        version: A versão do ML Lab.
    """

    status: str
    version: str


@router.get("/health")
def health() -> Health:
    """Se a API está a correr, e em que versão."""
    return Health(status="ok", version=__version__)
