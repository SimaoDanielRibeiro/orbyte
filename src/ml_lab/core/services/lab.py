"""O laboratório: todos os serviços de uma sessão, ligados entre si.

A interface guarda um ``Lab`` por sessão e pede-lhe tudo. Cada serviço continua a
poder ser usado sozinho (nos testes, por exemplo).
"""

from dataclasses import dataclass
from typing import Self

from ml_lab.config import Settings, get_settings
from ml_lab.core.services.datasets import DatasetService


@dataclass(frozen=True, slots=True)
class Lab:
    """Os serviços do ML Lab.

    Attributes:
        datasets: Carregar, explorar e preparar dados.
    """

    datasets: DatasetService

    @classmethod
    def create(cls, settings: Settings | None = None) -> Self:
        """Cria os serviços com as definições dadas (por omissão, as do processo)."""
        settings = settings or get_settings()
        return cls(datasets=DatasetService(settings))
