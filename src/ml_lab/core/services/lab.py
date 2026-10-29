"""O laboratório: todos os serviços de uma sessão, ligados entre si.

A interface guarda um ``Lab`` por sessão e pede-lhe tudo. Cada serviço continua a
poder ser usado sozinho (nos testes, por exemplo).
"""

from dataclasses import dataclass
from typing import Self

from ml_lab.config import Settings, get_settings
from ml_lab.core.foundation.errors import MLLabError
from ml_lab.core.foundation.events import EventBus, RunFinished
from ml_lab.core.services.analysis import AnalysisService
from ml_lab.core.services.automl import AutoMLService
from ml_lab.core.services.catalog import CatalogService
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import ExperimentService
from ml_lab.core.services.history import HistoryService
from ml_lab.core.services.laboratory import LaboratoryService
from ml_lab.core.services.packages import PackageService
from ml_lab.core.services.reinforcement import ReinforcementService
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
        analysis: O que se pergunta a um modelo depois do treino (a avaliação e o Explicar).
        laboratory: As demonstrações do ecrã Laboratório.
        history: O histórico de runs: registar, guardar, comparar e reproduzir.
        reinforcement: Treinar agentes de reinforcement learning (ecrã Reinforcement learning).
        automl: Os agentes do modo automático: procuram o melhor modelo para um pedido
            (ecrã Agentes).
        packages: Os pacotes dos modelos: guardar o que se preparou e aprendeu num ficheiro,
            a biblioteca e carregar um modelo treinado (ecrã Guardar).
    """

    datasets: DatasetService
    catalog: CatalogService
    experiments: ExperimentService
    analysis: AnalysisService
    laboratory: LaboratoryService
    history: HistoryService
    reinforcement: ReinforcementService
    automl: AutoMLService
    packages: PackageService

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
        # Um só EventBus: os treinos de modelos e os de agentes anunciam-se no mesmo sítio.
        bus = EventBus()
        datasets = DatasetService(settings)
        experiments = ExperimentService(datasets, bus=bus)
        reinforcement = ReinforcementService(settings, bus=bus)
        history = HistoryService(
            experiments,
            datasets,
            store if store is not None else SqliteStore(settings.database_path),
            ArtifactStore(settings.artifacts_dir / "runs"),
            reinforcement,
        )

        # O registo subscreve o fim de cada run (o padrão observer): cada treino que termina
        # fica registado, sem os serviços de treino saberem que o registo existe.
        def record(event: RunFinished) -> None:
            try:
                history.record(event.run_id)
            except MLLabError:  # o treino correu bem: um registo que falha não o estraga
                logger.exception("o run não ficou registado", extra={"run_id": event.run_id})

        bus.subscribe(RunFinished, record)
        return cls(
            datasets=datasets,
            catalog=CatalogService(settings),
            experiments=experiments,
            analysis=AnalysisService(experiments, datasets),
            laboratory=LaboratoryService(settings),
            history=history,
            reinforcement=reinforcement,
            automl=AutoMLService(datasets, experiments, history=history, settings=settings),
            packages=PackageService(experiments, history, settings),
        )
