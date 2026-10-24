"""O laboratório: todos os serviços de uma sessão, ligados entre si.

A interface guarda um ``Lab`` por sessão e pede-lhe tudo. Cada serviço continua a
poder ser usado sozinho (nos testes, por exemplo).
"""

from dataclasses import dataclass
from typing import Self

from ml_lab.config import Settings, get_settings
from ml_lab.core.services.analysis import AnalysisService
from ml_lab.core.services.catalog import CatalogService
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import ExperimentService
from ml_lab.core.services.laboratory import LaboratoryService


@dataclass(frozen=True, slots=True)
class Lab:
    """Os serviços do ML Lab.

    Attributes:
        datasets: Carregar, explorar e preparar dados.
        catalog: O catálogo de modelos.
        experiments: Treinar e avaliar (usa os datasets já carregados).
        analysis: O que se pergunta a um modelo depois do treino (Avaliar e Explicar).
        laboratory: As demonstrações do ecrã Laboratório.
    """

    datasets: DatasetService
    catalog: CatalogService
    experiments: ExperimentService
    analysis: AnalysisService
    laboratory: LaboratoryService

    @classmethod
    def create(cls, settings: Settings | None = None) -> Self:
        """Cria os serviços com as definições dadas (por omissão, as do processo)."""
        settings = settings or get_settings()
        datasets = DatasetService(settings)
        experiments = ExperimentService(datasets)
        return cls(
            datasets=datasets,
            catalog=CatalogService(settings),
            experiments=experiments,
            analysis=AnalysisService(experiments, datasets),
            laboratory=LaboratoryService(settings),
        )
