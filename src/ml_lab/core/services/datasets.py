"""Serviço de datasets: carregar, validar, perfilar e preparar dados.

É o que os ecrãs Dados, Explorar e Preparar usam. Recebe pedidos simples (o nome de
um exemplo, um ficheiro enviado, a configuração da preparação) e devolve
resultados prontos a mostrar, em estruturas simples: nunca DataFrames (ADR 0002).

Os datasets carregados ficam em memória, identificados pela chave da sua origem
(``SourceSpec.key``): carregar duas vezes a mesma origem não repete a leitura.
"""

import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from ml_lab.config import Settings, get_settings
from ml_lab.core.data.boundary import TablePreview, preview
from ml_lab.core.data.cleaning.base import CLEANING
from ml_lab.core.data.cleaning.suggestions import CleaningSuggestion, suggest_cleaning
from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import LoadedTable, SourceSpec, open_source
from ml_lab.core.data.io.builtin_source import SAMPLES
from ml_lab.core.data.io.file_source import FileSource
from ml_lab.core.data.io.synthetic_source import GENERATORS
from ml_lab.core.data.lineage import LineageStep
from ml_lab.core.data.preparation import (
    PreparationConfig,
    PreparedData,
    TaskOptions,
    prepare,
    prepare_folds,
    task_options,
)
from ml_lab.core.data.profiling.column_profile import ValueCount
from ml_lab.core.data.profiling.dataset_profile import DatasetProfile, profile_dataset
from ml_lab.core.data.profiling.type_inference import infer_types
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.task.task_detection import TaskSuggestion, suggest_task
from ml_lab.core.data.validation.checks import (
    MAX_FILE_BYTES,
    ValidationReport,
    file_format,
    merge_reports,
    validate_table,
)
from ml_lab.core.foundation.errors import DatasetNotFoundError, DataValidationError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.schemas.serialization import internal

MAX_PROJECTION_POINTS: Final = 3_000
"""Pontos mostrados no gráfico da projeção 2D; com mais, o gráfico fica lento."""

_SAFE_NAME: Final = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class SampleSummary:
    """Um dataset de exemplo, para a lista do ecrã Dados."""

    name: str
    title: str
    description: str
    target: str
    task_type: TaskType
    n_rows: int
    n_columns: int
    source: str


@dataclass(frozen=True, slots=True)
class GeneratorSummary:
    """Um gerador de dados sintéticos e os seus parâmetros (em JSON Schema)."""

    name: str
    title: str
    description: str
    task_type: TaskType
    parameters_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OperationSummary:
    """Uma operação de limpeza disponível."""

    name: str
    title: str
    description: str


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    """Uma coluna de um dataset carregado: o tipo do pandas e o tipo inferido."""

    name: str
    dtype: str
    kind: ColumnKind
    kind_reason: str


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Um dataset carregado.

    Attributes:
        id: Identificador para os outros pedidos ao serviço.
        name: Nome a mostrar.
        description: O que os dados descrevem.
        source: De onde vieram (para os voltar a obter).
        n_rows: Linhas.
        n_columns: Colunas.
        columns: As colunas e os seus tipos.
        suggested_target: A coluna-alvo habitual, se houver.
        validation: Os erros e avisos da validação.
    """

    id: str
    name: str
    description: str
    source: SourceSpec
    n_rows: int
    n_columns: int
    columns: tuple[ColumnInfo, ...]
    suggested_target: str | None
    validation: ValidationReport


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    """Uma parte da divisão: tamanho e, na classificação, a mistura de classes."""

    partition: Partition
    n_rows: int
    ratio: float
    classes: tuple[ValueCount, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectionPoints:
    """Os pontos da projeção 2D, prontos para um gráfico de dispersão.

    Attributes:
        method: ``"pca"`` ou ``"columns"``.
        axis_labels: Nome de cada eixo.
        x: Coordenada horizontal de cada ponto.
        y: Coordenada vertical de cada ponto.
        partitions: A parte (treino, validação, teste) de cada ponto.
        labels: Na classificação, a classe de cada ponto.
        values: Na regressão, o valor do alvo de cada ponto.
        explained_variance_ratio: Na PCA, a variância explicada por cada eixo.
        loadings: Na PCA, o peso de cada feature em cada eixo.
        feature_names: As features a que os pesos se referem.
    """

    method: str
    axis_labels: tuple[str, str]
    x: tuple[float, ...]
    y: tuple[float, ...]
    partitions: tuple[Partition, ...]
    labels: tuple[str, ...] = ()
    values: tuple[float, ...] = ()
    explained_variance_ratio: tuple[float, float] | None = None
    loadings: tuple[tuple[float, ...], ...] | None = None
    feature_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    """O resultado da preparação, pronto a mostrar no ecrã Preparar.

    Attributes:
        dataset_id: O dataset preparado.
        task: A sugestão de tarefa para o alvo.
        task_type: A tarefa usada.
        lineage: Todos os passos, da entrada à saída.
        partitions: O tamanho (e as classes) de cada parte.
        feature_names: As features que chegam ao modelo, depois das transformações.
        class_labels: As classes, na classificação.
        cleaned_preview: As primeiras linhas depois da limpeza.
        transformed_preview: As primeiras linhas de treino depois das transformações.
        projection: Os pontos da projeção 2D.
        notices: Os avisos de todas as etapas.
    """

    dataset_id: str
    task: TaskSuggestion
    task_type: TaskType
    lineage: tuple[LineageStep, ...]
    partitions: tuple[PartitionSummary, ...]
    feature_names: tuple[str, ...]
    class_labels: tuple[str, ...]
    cleaned_preview: TablePreview
    transformed_preview: TablePreview
    projection: ProjectionPoints | None
    notices: tuple[Notice, ...] = ()


@dataclass(slots=True)
class _Entry:
    table: LoadedTable
    summary: DatasetSummary
    profiles: dict[str | None, DatasetProfile] = field(default_factory=dict)
    prepared: dict[str, PreparedData] = field(default_factory=dict)


class DatasetService:
    """Casos de uso dos dados: o que a interface pode pedir sobre datasets.

    Args:
        settings: As definições (pastas de exemplos e de uploads); por omissão, as do processo.
        max_datasets: Quantos datasets ficam em memória; os mais antigos saem primeiro.
    """

    def __init__(self, settings: Settings | None = None, *, max_datasets: int = 20) -> None:
        self._settings = settings or get_settings()
        self._max_datasets = max_datasets
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()

    # --- catálogos -------------------------------------------------------------------

    def samples(self) -> list[SampleSummary]:
        """Os datasets de exemplo."""
        return [
            SampleSummary(
                name=info.name,
                title=info.title,
                description=info.description,
                target=info.target,
                task_type=info.task_type,
                n_rows=info.n_rows,
                n_columns=info.n_columns,
                source=info.source,
            )
            for _, info in SAMPLES.items()
        ]

    def generators(self) -> list[GeneratorSummary]:
        """Os geradores de dados sintéticos, com os parâmetros de cada um."""
        return [
            GeneratorSummary(
                info.name, info.title, info.description, info.task_type, info.parameters_schema()
            )
            for _, info in GENERATORS.items()
        ]

    def cleaning_operations(self) -> list[OperationSummary]:
        """As operações de limpeza disponíveis."""
        return [OperationSummary(op.name, op.title, op.description) for _, op in CLEANING.items()]

    # --- carregar --------------------------------------------------------------------

    def load(self, spec: SourceSpec) -> DatasetSummary:
        """Carrega os dados descritos por ``spec`` (ou reaproveita-os, se já estiverem carregados).

        Raises:
            DataSourceError: Se não for possível ler a origem.
            DataValidationError: Se os dados não forem utilizáveis.
        """
        with self._lock:
            entry = self._entries.get(spec.key)
            if entry is not None:
                self._entries.move_to_end(spec.key)
                return entry.summary
        table = open_source(spec).load()
        summary = _summarize(spec.key, table)
        with self._lock:
            self._entries[spec.key] = _Entry(table, summary)
            while len(self._entries) > self._max_datasets:
                self._entries.popitem(last=False)
        return summary

    def load_sample(self, name: str) -> DatasetSummary:
        """Carrega um dataset de exemplo."""
        return self.load(SourceSpec("sample", {"name": name}))

    def generate(
        self, generator: str, parameters: Mapping[str, float], seed: int
    ) -> DatasetSummary:
        """Gera um dataset sintético."""
        spec = SourceSpec(
            "synthetic", {"generator": generator, "seed": seed, "parameters": dict(parameters)}
        )
        return self.load(open_source(spec).spec())

    def upload(self, filename: str, content: bytes) -> DatasetSummary:
        """Guarda um ficheiro enviado pelo utilizador em ``data/uploads/`` e carrega-o.

        O nome gravado leva o início do SHA-256 do conteúdo: enviar outro ficheiro com o
        mesmo nome não apaga o primeiro, e reenviar o mesmo ficheiro não o duplica.

        Raises:
            DataSourceError: Se o formato não for suportado.
            DataValidationError: Se o ficheiro estiver vazio, for grande demais ou ilegível.
        """
        original = Path(filename).name
        file_format(Path(original))  # recusa formatos não suportados antes de gravar
        if not content:
            raise DataValidationError([f"The file '{original}' is empty (0 bytes)."])
        if len(content) > MAX_FILE_BYTES:
            raise DataValidationError([f"The file '{original}' is too big."])
        suffixes = "".join(Path(original).suffixes).lower()
        stem = _SAFE_NAME.sub("_", original[: len(original) - len(suffixes)]).strip("_") or "data"
        digest = hashlib.sha256(content).hexdigest()[:8]
        directory = self._settings.uploads_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}-{digest}{suffixes}"
        if not path.exists():
            path.write_bytes(content)
        # Dentro do projeto, o caminho fica relativo à raiz (a mesma que o FileSource usa
        # para o ler), para a experiência continuar válida noutro computador.
        root = get_settings().home
        stored = path.relative_to(root) if path.is_relative_to(root) else path
        return self.load(FileSource(stored.as_posix(), name=original).spec())

    # --- consultar -------------------------------------------------------------------

    def summary(self, dataset_id: str) -> DatasetSummary:
        """O resumo de um dataset carregado."""
        return self._entry(dataset_id).summary

    def loaded(self) -> list[DatasetSummary]:
        """Os datasets carregados, do mais recente para o mais antigo."""
        with self._lock:
            return [entry.summary for entry in reversed(self._entries.values())]

    def preview(self, dataset_id: str, n_rows: int = 20) -> TablePreview:
        """As primeiras linhas, tal como foram lidas."""
        return preview(self._entry(dataset_id).table.frame, n_rows)

    def profile(self, dataset_id: str, target: str | None = None) -> DatasetProfile:
        """O perfil do dataset (com o equilíbrio das classes, se houver alvo)."""
        entry = self._entry(dataset_id)
        with self._lock:
            cached = entry.profiles.get(target)
        if cached is not None:
            return cached
        profile = profile_dataset(entry.table.frame, target=target)
        with self._lock:
            entry.profiles[target] = profile
        return profile

    def suggest_task(self, dataset_id: str, target: str | None) -> TaskSuggestion:
        """O tipo de tarefa sugerido para o alvo."""
        return suggest_task(self._entry(dataset_id).table.frame, target)

    def suggest_cleaning(
        self, dataset_id: str, target: str | None = None
    ) -> tuple[CleaningSuggestion, ...]:
        """Uma receita de limpeza sugerida a partir do perfil."""
        return suggest_cleaning(self.profile(dataset_id), target)

    def task_options(self, task_type: TaskType, target: str | None) -> TaskOptions:
        """O que muda com a tarefa escolhida: a coluna de referência e as divisões possíveis."""
        return task_options(task_type, target)

    # --- preparar --------------------------------------------------------------------

    @internal
    def prepared(self, spec: SourceSpec, config: PreparationConfig, seed: int) -> PreparedData:
        """Os dados preparados, para os outros serviços (com ``Dataset`` e DataFrames).

        Carrega a origem se for preciso; com a mesma configuração e a mesma seed,
        reaproveita a preparação anterior.
        """
        summary = self.load(spec)
        entry = self._entry(summary.id)
        key = _config_key(config, seed)
        with self._lock:
            cached = entry.prepared.get(key)
        if cached is not None:
            return cached
        result = prepare(entry.table, config, seeds=SeedManager(seed))
        with self._lock:
            entry.prepared = {key: result}  # só a última: as preparações ocupam memória
        return result

    @internal
    def folds(
        self,
        spec: SourceSpec,
        config: PreparationConfig,
        seed: int,
        k: int,
        max_rows: int | None = None,
    ) -> tuple[Dataset, ...]:
        """Os dados preparados para a validação cruzada: um ``Dataset`` por fold.

        O teste da divisão normal fica de fora; o treino parte-se em ``k`` folds, e as
        transformações aprendem de novo em cada um. Com ``max_rows``, os folds usam só uma
        amostra do treino com esse número de linhas (ver ``prepare_folds``).
        """
        summary = self.load(spec)
        table = self._entry(summary.id).table
        return prepare_folds(table, config, seeds=SeedManager(seed), k=k, max_rows=max_rows)

    def prepare(self, dataset_id: str, config: PreparationConfig, seed: int) -> PreparationSummary:
        """Prepara o dataset e resume o resultado para o ecrã Preparar.

        Raises:
            DataError: Se os dados não servirem para a configuração pedida.
            InvalidParameterError: Se a configuração não fizer sentido.
        """
        result = self.prepared(self.summary(dataset_id).source, config, seed)
        dataset = result.dataset
        target = config.target
        partitions = []
        total = sum(len(split) for split in dataset.splits.values())
        for part, split in dataset.splits.items():
            classes: tuple[ValueCount, ...] = ()
            if dataset.class_labels and split.y is not None:
                counts = np.bincount(split.y.astype(np.int64), minlength=dataset.n_classes)
                classes = tuple(
                    ValueCount(label, int(count), int(count) / len(split))
                    for label, count in zip(dataset.class_labels, counts, strict=True)
                )
            partitions.append(PartitionSummary(part, len(split), len(split) / total, classes))
        return PreparationSummary(
            dataset_id=dataset_id,
            task=result.task,
            task_type=dataset.task_type,
            lineage=dataset.lineage.steps,
            partitions=tuple(partitions),
            feature_names=dataset.feature_names,
            class_labels=dataset.class_labels,
            cleaned_preview=preview(result.cleaned),
            transformed_preview=preview(result.transformed[Partition.TRAIN]),
            projection=_projection_points(result, target),
            notices=result.notices,
        )

    @internal
    def table(self, dataset_id: str) -> LoadedTable:
        """A tabela carregada (para outros serviços; a interface usa ``preview``)."""
        return self._entry(dataset_id).table

    def _entry(self, dataset_id: str) -> _Entry:
        with self._lock:
            entry = self._entries.get(dataset_id)
        if entry is None:
            raise DatasetNotFoundError(dataset_id)
        return entry


def _summarize(dataset_id: str, table: LoadedTable) -> DatasetSummary:
    frame = table.frame
    inferences = infer_types(frame)
    columns = tuple(
        ColumnInfo(
            str(name),
            str(frame[name].dtype),
            inferences[str(name)].kind,
            inferences[str(name)].reason,
        )
        for name in frame.columns
    )
    return DatasetSummary(
        id=dataset_id,
        name=table.name,
        description=table.description,
        source=table.spec,
        n_rows=len(frame),
        n_columns=frame.shape[1],
        columns=columns,
        suggested_target=table.suggested_target,
        validation=merge_reports(table.notices, validate_table(frame)),
    )


def _config_key(config: PreparationConfig, seed: int) -> str:
    payload = {"config": asdict(config), "seed": seed}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _projection_points(result: PreparedData, target: str | None) -> ProjectionPoints | None:
    dataset = result.dataset
    projection = dataset.projection
    if projection is None:
        return None
    xs: list[float] = []
    ys: list[float] = []
    parts: list[Partition] = []
    labels: list[str] = []
    values: list[float] = []
    total = sum(len(split) for split in dataset.splits.values())
    # Com muitos pontos, mostra-se uma linha em cada ``step``: o gráfico fica leve e a
    # forma dos dados continua igual.
    step = max(1, -(-total // MAX_PROJECTION_POINTS))
    for part, split in dataset.splits.items():
        coordinates = projection.coordinates[part][::step]
        xs.extend(float(v) for v in coordinates[:, 0])
        ys.extend(float(v) for v in coordinates[:, 1])
        parts.extend([part] * len(coordinates))
        if split.y is not None and dataset.class_labels:
            labels.extend(dataset.class_labels[int(code)] for code in split.y[::step])
        elif split.y is not None and target is not None:
            values.extend(float(v) for v in split.y[::step])
    loadings = None
    if projection.loadings is not None:
        loadings = tuple(tuple(float(w) for w in row) for row in projection.loadings)
    return ProjectionPoints(
        method=projection.method,
        axis_labels=projection.axis_labels,
        x=tuple(xs),
        y=tuple(ys),
        partitions=tuple(parts),
        labels=tuple(labels),
        values=tuple(values),
        explained_variance_ratio=projection.explained_variance_ratio,
        loadings=loadings,
        feature_names=projection.feature_names,
    )
