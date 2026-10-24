"""O laboratório: todos os serviços de uma sessão, ligados entre si.

A interface guarda um ``Lab`` por sessão e pede-lhe tudo. Cada serviço continua a
poder ser usado sozinho (nos testes, por exemplo).
"""

from dataclasses import dataclass
from typing import Self

from ml_lab.config import Settings, get_settings
from ml_lab.core.foundation.errors import MLLabError
from ml_lab.core.foundation.events import RunFinished
from ml_lab.core.services.analysis import AnalysisService
from ml_lab.core.services.catalog import CatalogService
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import ExperimentService
from ml_lab.core.services.history import HistoryService
from ml_lab.core.services.laboratory import LaboratoryService
from ml_lab.core.tracking.artifacts import ArtifactStore
from ml_lab.core.tracking.sqlite_store import SqliteStore
from ml_lab.core.tracking.store import ExperimentStore
from ml_lab.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Lab:
    """Os serviços do ML Lab.

    Attributes:
        datasets: Carregar, explorar e preparar dados.
        catalog: O catálogo de modelos.
        experiments: Treinar e avaliar (usa os datasets já carregados).
        analysis: O que se pergunta a um modelo depois do treino (Avaliar e Explicar).
        laboratory: As demonstrações do ecrã Laboratório.
        history: O histórico de runs: registar, guardar, comparar e reproduzir.
    """

    datasets: DatasetService
    catalog: CatalogService
    experiments: ExperimentService
    analysis: AnalysisService
    laboratory: LaboratoryService
    history: HistoryService

    @classmethod
    def create(
        cls, settings: Settings | None = None, *, store: ExperimentStore | None = None
    ) -> Self:
        """Cria os serviços com as definições dadas (por omissão, as do processo).

        Args:
            settings: As definições (pastas e base de dados).
            store: O registo de experiências; por omissão, o SQLite de ``settings``.
        """
        settings = settings or get_settings()
        datasets = DatasetService(settings)
        experiments = ExperimentService(datasets)
        history = HistoryService(
            experiments,
            datasets,
            store if store is not None else SqliteStore(settings.database_path),
            ArtifactStore(settings.artifacts_dir / "runs"),
        )

        # O registo subscreve o fim de cada run (o padrão observer): cada treino que termina
        # fica registado, sem o serviço de experiências saber que o registo existe.
        def record(event: RunFinished) -> None:
            try:
                history.record(event.run_id)
            except MLLabError:  # o treino correu bem: um registo que falha não o estraga
                logger.exception("o run não ficou registado", extra={"run_id": event.run_id})

        experiments.bus.subscribe(RunFinished, record)
        return cls(
            datasets=datasets,
            catalog=CatalogService(settings),
            experiments=experiments,
            analysis=AnalysisService(experiments, datasets),
            laboratory=LaboratoryService(settings),
            history=history,
        )
