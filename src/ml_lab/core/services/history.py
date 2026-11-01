"""Serviço do histórico: registar, guardar, abrir, comparar e reproduzir runs (ecrã Histórico).

A memória do ML Lab tem dois passos (ADR 0020):

1. **registar**: cada run que termina entra sozinho no registo, com a configuração, os
   relatórios, a curva da loss e o que o modelo aprendeu (o resultado em JSON, sem o
   treino passo a passo). É leve, e chega para comparar runs e para os reproduzir;
2. **guardar**: a pedido, juntam-se o modelo treinado e o treino passo a passo (a
   animação e, numa rede neuronal, as camadas em cada snapshot), que é o que mais pesa.
   Um run guardado volta a abrir com tudo, mesmo depois de fechar a app: o modelo vem do
   ficheiro e os dados preparam-se outra vez (a mesma configuração e a mesma seed dão os
   mesmos dados), por isso os gráficos da avaliação e o Explicar voltam a funcionar.

Os runs de reinforcement learning (fase 8) entram no mesmo histórico, com a tarefa
``CONTROL``: o ambiente fica onde fica o dataset, e o agente onde fica o modelo. O que muda
é o documento (um ``ReinforcementResult``), que se abre com ``open_reinforcement``, e o que se
guarda a pedido (o agente e o que ele sabia em cada snapshot).
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.foundation.errors import InvalidParameterError, MLLabError, RunNotInMemoryError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.schemas.experiment import ExperimentConfig
from ml_lab.core.schemas.reinforcement import ReinforcementConfig
from ml_lab.core.schemas.serialization import from_json, internal, to_json
from ml_lab.core.services.comparison import MAX_COMPARED, RunComparison, compare_runs
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import ExperimentResult, ExperimentService, RunArtifacts
from ml_lab.core.services.reinforcement import ReinforcementResult, ReinforcementService
from ml_lab.core.tracking.artifacts import ArtifactStore
from ml_lab.core.tracking.store import ExperimentStore, RunEntry
from ml_lab.core.training.jobs import TrainingJob
from ml_lab.core.visualization_data.animation import TrainingAnimation
from ml_lab.core.visualization_data.network import NetworkTrace
from ml_lab.core.visualization_data.policy import PolicyTrace

type RunResult = ExperimentResult | ReinforcementResult
"""O resultado de um run, de um modelo (com um dataset) ou de um agente (num ambiente)."""


class HistoryService:
    """O histórico de runs.

    Args:
        experiments: Onde estão o modelo e o resultado dos últimos runs desta sessão.
        datasets: Para voltar a preparar os dados de um run guardado que se reabre.
        store: O registo (em SQLite, na app; em memória, nos testes).
        files: Os ficheiros dos runs guardados (o modelo e o treino passo a passo).
        reinforcement: Onde estão o agente e o resultado dos últimos treinos de
            reinforcement learning desta sessão.
    """

    def __init__(
        self,
        experiments: ExperimentService,
        datasets: DatasetService,
        store: ExperimentStore,
        files: ArtifactStore,
        reinforcement: ReinforcementService,
    ) -> None:
        self._experiments = experiments
        self._datasets = datasets
        self._store = store
        self._files = files
        self._reinforcement = reinforcement

    @internal
    def record(self, run_id: str) -> RunEntry:
        """Regista um run que acabou de terminar (o ``Lab`` liga-o ao fim de cada run).

        Raises:
            RunNotInMemoryError: Se o run já não estiver em memória.
        """
        _, result = self._session(run_id)
        entry = _entry(result)
        # O treino passo a passo fica de fora: é o que mais pesa, e só se guarda a pedido.
        self._store.add(entry, to_json(_light(result)))
        return entry

    @internal
    def adopt(
        self,
        result: ExperimentResult,
        model: Any | None,
        documents: Mapping[str, Any],
        *,
        name: str,
    ) -> RunEntry:
        """Põe no histórico um run que veio de um pacote (ecrã Guardar, ADR 0039).

        Com o modelo, o run fica guardado (com o treino passo a passo que o pacote trouxer) e
        volta a abrir com tudo; sem ele (um pacote "só a ficha"), fica só registado.

        Args:
            result: O run, como o pacote o trouxe.
            model: O modelo treinado, ou ``None``.
            documents: O treino passo a passo, pelo nome do ficheiro (``{"animation": ...}``).
            name: O nome do run no histórico.
        """
        entry = replace(_entry(result), name=_name(name), saved=model is not None)
        self._store.add(entry, to_json(_light(result)))
        if model is not None:
            self._files.save(result.run_id, model, documents)
        return entry

    def runs(
        self,
        *,
        task_type: TaskType | None = None,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> tuple[RunEntry, ...]:
        """O histórico, do run mais recente para o mais antigo (filtrado, se se pedir)."""
        return tuple(
            entry
            for entry in self._store.entries()
            if (task_type is None or entry.task_type is task_type)
            and (dataset_name is None or entry.dataset_name == dataset_name)
            and (model_name is None or entry.model_name == model_name)
        )

    def entry(self, run_id: str) -> RunEntry:
        """A entrada de um run do histórico.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
        """
        return self._store.entry(run_id)

    def save(self, run_id: str, name: str | None = None) -> RunEntry:
        """Guarda o modelo (ou o agente) treinado e o treino passo a passo de um run desta sessão.

        Args:
            run_id: O run.
            name: Um nome novo para o reconhecer no histórico (opcional).

        Raises:
            RunNotInMemoryError: Se o run já não estiver em memória (só os últimos ficam).
            RunNotFoundError: Se o run não estiver no registo.
        """
        trained, result = self._session(run_id)
        self._store.entry(run_id)  # confirma que o run está no registo antes de gravar ficheiros
        self._files.save(run_id, trained, _step_by_step(result))
        return self._store.update(run_id, saved=True, name=None if name is None else _name(name))

    def rename(self, run_id: str, name: str) -> RunEntry:
        """Muda o nome de um run.

        Raises:
            InvalidParameterError: Se o nome estiver vazio.
            RunNotFoundError: Se o run não estiver no registo.
        """
        return self._store.update(run_id, name=_name(name))

    def delete(self, run_id: str) -> None:
        """Apaga um run do histórico, com os seus ficheiros."""
        self._store.delete(run_id)
        self._files.delete(run_id)

    def open(self, run_id: str) -> ExperimentResult:
        """O resultado de um run, pronto a mostrar nos ecrãs Treinar e avaliar e Explicar.

        Num run guardado, o modelo volta do ficheiro e os dados preparam-se outra vez. Se
        isso falhar (por exemplo, o ficheiro de dados já não existe), o resultado abre na
        mesma, com um aviso, e só os gráficos que precisam do modelo ficam de fora.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
            InvalidParameterError: Se o run for de reinforcement learning (abre-se com
                ``open_reinforcement``).
            TrainingInProgressError: Se o run ainda estiver a treinar.
            TrainingFailedError: Se o treino do run falhou (e por isso não ficou registado).
        """
        try:
            return self._experiments.result(run_id)  # desta sessão: está tudo em memória
        except RunNotInMemoryError:
            pass
        entry = self._store.entry(run_id)
        if entry.task_type is TaskType.CONTROL:
            raise InvalidParameterError(
                f"O run '{run_id}' é de reinforcement learning: abre-o no ecrã Reinforcement "
                "learning."
            )
        result = from_json(ExperimentResult, self._store.document(run_id))
        if not entry.saved:
            return result
        try:
            return self._restore(result)
        except MLLabError as error:
            notice = Notice.warning(
                "not_restored",
                f"O modelo guardado não voltou a abrir ({error}). Os gráficos da avaliação e o "
                "Explicar ficam indisponíveis para este run; o resto está como foi registado.",
            )
            return replace(result, notices=(notice, *result.notices))

    def open_reinforcement(self, run_id: str) -> ReinforcementResult:
        """O resultado de um run de reinforcement learning, pronto a mostrar.

        Num run guardado, volta também o que o agente sabia em cada snapshot.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
            InvalidParameterError: Se o run não for de reinforcement learning.
            TrainingInProgressError: Se o run ainda estiver a treinar.
            TrainingFailedError: Se o treino do run falhou.
        """
        try:
            return self._reinforcement.result(run_id)  # desta sessão: está tudo em memória
        except RunNotInMemoryError:
            pass
        entry = self._store.entry(run_id)
        if entry.task_type is not TaskType.CONTROL:
            raise InvalidParameterError(
                f"O run '{run_id}' não é de reinforcement learning: abre-o no ecrã Histórico."
            )
        result = from_json(ReinforcementResult, self._store.document(run_id))
        trace = self._files.document(run_id, "trace") if entry.saved else None
        return result if trace is None else replace(result, trace=from_json(PolicyTrace, trace))

    def compare(self, run_ids: Sequence[str]) -> RunComparison:
        """Os runs lado a lado: as métricas, as escolhas que mudam e as curvas da loss.

        Raises:
            InvalidParameterError: Se não forem entre 2 e ``MAX_COMPARED`` runs diferentes.
            RunNotFoundError: Se algum run não estiver no registo.
        """
        chosen = list(dict.fromkeys(run_ids))
        if not 2 <= len(chosen) <= MAX_COMPARED:
            raise InvalidParameterError(
                f"Escolhe entre 2 e {MAX_COMPARED} runs diferentes para comparar."
            )
        entries = [self._store.entry(run_id) for run_id in chosen]
        return compare_runs(entries, [self._document(entry) for entry in entries])

    def reproduce(self, run_id: str) -> TrainingJob[Any]:
        """Volta a treinar um run com a mesma configuração e a mesma seed, em segundo plano.

        O run novo entra no histórico como qualquer outro; comparado com o original, deve
        ter as mesmas métricas.

        Returns:
            O treino: de uma experiência, ou de um agente (num run de reinforcement learning).

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
        """
        entry = self._store.entry(run_id)
        config = self._store.document(run_id)["config"]
        if entry.task_type is TaskType.CONTROL:
            return self._reinforcement.start(from_json(ReinforcementConfig, config))
        return self._experiments.start(from_json(ExperimentConfig, config))

    def _session(self, run_id: str) -> tuple[Any, RunResult]:
        """O modelo (ou o agente) e o resultado de um run desta sessão.

        Raises:
            RunNotInMemoryError: Se nenhum serviço o tiver em memória.
        """
        try:
            run = self._experiments.artifacts(run_id)
            return run.model, run.result
        except RunNotInMemoryError:
            trained = self._reinforcement.artifacts(run_id)
            return trained.agent, trained.result

    def _document(self, entry: RunEntry) -> RunResult:
        """O documento de um run do registo, do tipo certo para a sua tarefa."""
        document = self._store.document(entry.run_id)
        if entry.task_type is TaskType.CONTROL:
            return from_json(ReinforcementResult, document)
        return from_json(ExperimentResult, document)

    def _restore(self, result: ExperimentResult) -> ExperimentResult:
        """Volta a pôr em memória o modelo e os dados de um run guardado (e o passo a passo)."""
        animation = self._files.document(result.run_id, "animation")
        network = self._files.document(result.run_id, "network")
        result = replace(
            result,
            animation=None if animation is None else from_json(TrainingAnimation, animation),
            network=None if network is None else from_json(NetworkTrace, network),
        )
        model = self._files.model(result.run_id)
        config = result.config
        dataset = self._datasets.prepared(config.dataset, config.preparation, result.seed).dataset
        self._experiments.remember(RunArtifacts(model, dataset, config, result))
        return result


def _entry(result: RunResult) -> RunEntry:
    report = result.main_report
    primary = result.primary
    return RunEntry(
        run_id=result.run_id,
        created_at=datetime.now(UTC),
        name=f"{result.model_display_name} em {result.dataset_name}",
        dataset_name=result.dataset_name,
        task_type=result.task_type,
        model_name=result.model_name,
        model_display_name=result.model_display_name,
        seed=result.seed,
        partition=report.partition,
        primary_metric=PRIMARY_METRIC.get(result.task_type),
        metrics=report.values(),
        primary_label=None if primary is None else primary.display_name,
        primary_as_percent=primary is not None and primary.as_percent,
    )


def _light(result: RunResult) -> RunResult:
    """O resultado sem o treino passo a passo (o que se regista sempre)."""
    if isinstance(result, ReinforcementResult):
        return replace(result, trace=None)
    return replace(result, animation=None, network=None)


def _step_by_step(result: RunResult) -> dict[str, Any]:
    """O treino passo a passo de um run, em valores de JSON, pelo nome do seu ficheiro."""
    parts: dict[str, Any] = (
        {"trace": result.trace}
        if isinstance(result, ReinforcementResult)
        else {"animation": result.animation, "network": result.network}
    )
    return {name: to_json(part) for name, part in parts.items() if part is not None}


def _name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise InvalidParameterError("O nome do run não pode ficar vazio.")
    return cleaned
