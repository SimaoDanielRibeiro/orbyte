"""Serviço do histórico: registar, guardar, abrir, comparar e reproduzir runs (ecrã Experiências).

A memória do ML Lab tem dois passos (ADR 0020):

1. **registar**: cada run que termina entra sozinho no registo, com a configuração, os
   relatórios, a curva da loss e o que o modelo aprendeu (o resultado em JSON, sem o
   treino passo a passo). É leve, e chega para comparar runs e para os reproduzir;
2. **guardar**: a pedido, juntam-se o modelo treinado e o treino passo a passo (a
   animação e, numa rede neuronal, as camadas em cada snapshot), que é o que mais pesa.
   Um run guardado volta a abrir com tudo, mesmo depois de fechar a app: o modelo vem do
   ficheiro e os dados preparam-se outra vez (a mesma configuração e a mesma seed dão os
   mesmos dados), por isso os gráficos do Avaliar e o Explicar voltam a funcionar.
"""

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.foundation.errors import InvalidParameterError, MLLabError, ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.schemas.experiment import ExperimentConfig
from ml_lab.core.schemas.serialization import from_json, internal, to_json
from ml_lab.core.services.comparison import MAX_COMPARED, RunComparison, compare_runs
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.services.experiments import ExperimentResult, ExperimentService, RunArtifacts
from ml_lab.core.tracking.artifacts import ArtifactStore
from ml_lab.core.tracking.store import ExperimentStore, RunEntry
from ml_lab.core.training.jobs import TrainingJob
from ml_lab.core.visualization_data.animation import TrainingAnimation
from ml_lab.core.visualization_data.network import NetworkTrace


class HistoryService:
    """O histórico de runs.

    Args:
        experiments: Onde estão o modelo e o resultado dos últimos runs desta sessão.
        datasets: Para voltar a preparar os dados de um run guardado que se reabre.
        store: O registo (em SQLite, na app; em memória, nos testes).
        files: Os ficheiros dos runs guardados (o modelo e o treino passo a passo).
    """

    def __init__(
        self,
        experiments: ExperimentService,
        datasets: DatasetService,
        store: ExperimentStore,
        files: ArtifactStore,
    ) -> None:
        self._experiments = experiments
        self._datasets = datasets
        self._store = store
        self._files = files

    @internal
    def record(self, run_id: str) -> RunEntry:
        """Regista um run que acabou de terminar (o ``Lab`` liga-o ao fim de cada run).

        Raises:
            ModelError: Se o run já não estiver em memória.
        """
        result = self._experiments.artifacts(run_id).result
        entry = _entry(result)
        # O treino passo a passo fica de fora: é o que mais pesa, e só se guarda a pedido.
        self._store.add(entry, to_json(replace(result, animation=None, network=None)))
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
        """Guarda o modelo treinado e o treino passo a passo de um run desta sessão.

        Args:
            run_id: O run.
            name: Um nome novo para o reconhecer no histórico (opcional).

        Raises:
            ModelError: Se o run já não estiver em memória (só os últimos ficam).
            RunNotFoundError: Se o run não estiver no registo.
        """
        artifacts = self._experiments.artifacts(run_id)
        self._store.entry(run_id)  # confirma que o run está no registo antes de gravar ficheiros
        self._files.save(run_id, artifacts.model, _step_by_step(artifacts.result))
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
        """O resultado de um run, pronto a mostrar nos ecrãs Treinar, Avaliar e Explicar.

        Num run guardado, o modelo volta do ficheiro e os dados preparam-se outra vez. Se
        isso falhar (por exemplo, o ficheiro de dados já não existe), o resultado abre na
        mesma, com um aviso, e só os gráficos que precisam do modelo ficam de fora.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
        """
        try:
            return self._experiments.artifacts(run_id).result  # desta sessão: está tudo em memória
        except ModelError:
            pass
        entry = self._store.entry(run_id)
        result = from_json(ExperimentResult, self._store.document(run_id))
        if not entry.saved:
            return result
        try:
            return self._restore(result)
        except MLLabError as error:
            notice = Notice.warning(
                "not_restored",
                f"O modelo guardado não voltou a abrir ({error}). Os gráficos do Avaliar e o "
                "Explicar ficam indisponíveis para este run; o resto está como foi registado.",
            )
            return replace(result, notices=(notice, *result.notices))

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
        results = [from_json(ExperimentResult, self._store.document(run_id)) for run_id in chosen]
        return compare_runs(entries, results)

    def reproduce(self, run_id: str) -> TrainingJob[ExperimentResult]:
        """Volta a treinar um run com a mesma configuração e a mesma seed, em segundo plano.

        O run novo entra no histórico como qualquer outro; comparado com o original, deve
        ter as mesmas métricas.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
        """
        config = from_json(ExperimentConfig, self._store.document(run_id)["config"])
        return self._experiments.start(config)

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


def _entry(result: ExperimentResult) -> RunEntry:
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


def _step_by_step(result: ExperimentResult) -> dict[str, Any]:
    """O treino passo a passo de um run, em valores de JSON, pelo nome do seu ficheiro."""
    parts = {"animation": result.animation, "network": result.network}
    return {name: to_json(part) for name, part in parts.items() if part is not None}


def _name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise InvalidParameterError("O nome do run não pode ficar vazio.")
    return cleaned
