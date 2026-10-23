"""Preparação: as etapas 4 a 9 da camada de dados, pela ordem certa.

Recebe a tabela carregada e uma configuração declarativa, e devolve o ``Dataset``
pronto para os modelos:

4. **limpeza** (a receita de passos), que não aprende nada com os dados;
5. **tarefa**: escolhe o alvo, confirma o tipo de tarefa e remove as linhas sem alvo;
6. **divisão** em treino, validação e teste, com a seed;
7. **transformação**, aprendida só com o treino e aplicada a todas as partes;
8. **projeção** 2D para os gráficos;
9. **saída**: matrizes NumPy, alvo codificado e linhagem completa.

A mesma tabela, a mesma configuração e a mesma seed dão sempre o mesmo ``Dataset``.
"""

from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from ml_lab.core.data.boundary import class_labels, target_values, to_matrix
from ml_lab.core.data.cleaning.base import CleaningStep, apply_cleaning
from ml_lab.core.data.dataset import Dataset, DataSplit
from ml_lab.core.data.io.base import LoadedTable
from ml_lab.core.data.lineage import Lineage, LineageStep, Stage
from ml_lab.core.data.profiling.type_inference import TypeInference, infer_types, schema_from
from ml_lab.core.data.projection.projection_2d import ProjectionConfig, project_2d
from ml_lab.core.data.schema import ColumnRole, TableSchema
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy, split_rows
from ml_lab.core.data.task.task_detection import TaskSuggestion, suggest_task
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig, build_pipeline
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.formatting import format_count, format_quantity

SPLIT_SEED: Final = "split"
"""Nome do componente da divisão: a sua seed é ``seeds.seed_for("split")`` (ADR 0006)."""


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    """Tudo o que é preciso para ir da tabela carregada ao ``Dataset``.

    Attributes:
        target: A coluna-alvo, ou ``None`` para aprendizagem não supervisionada.
        task_type: O tipo de tarefa; ``None`` aceita a sugestão automática.
        cleaning: A receita de limpeza.
        split: Como dividir.
        transforms: Que transformações aplicar.
        projection: Como obter a versão 2D para os gráficos.
    """

    target: str | None = None
    task_type: TaskType | None = None
    cleaning: tuple[CleaningStep, ...] = ()
    split: SplitConfig = field(default_factory=SplitConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    def __post_init__(self) -> None:
        if self.task_type is not None:
            object.__setattr__(self, "task_type", TaskType(self.task_type))
        object.__setattr__(self, "cleaning", tuple(self.cleaning))


@dataclass(frozen=True, slots=True, eq=False)
class PreparedData:
    """O resultado da preparação.

    Attributes:
        dataset: Os dados prontos para os modelos.
        cleaned: A tabela depois da limpeza e da remoção das linhas sem alvo.
        transformed: As features de cada parte depois das transformações, como tabelas.
        task: A sugestão de tarefa para o alvo escolhido.
        notices: Os avisos de todas as etapas.
    """

    dataset: Dataset
    cleaned: pd.DataFrame
    transformed: dict[Partition, pd.DataFrame]
    task: TaskSuggestion
    notices: tuple[Notice, ...] = ()


def prepare(table: LoadedTable, config: PreparationConfig, *, seeds: SeedManager) -> PreparedData:
    """Prepara os dados de ``table`` segundo ``config``.

    Args:
        table: A tabela carregada; não é alterada.
        config: A configuração da preparação.
        seeds: As seeds da experiência; a divisão usa ``seeds.seed_for("split")``.

    Raises:
        DataError: Se os dados não servirem (por exemplo, sem linhas depois da limpeza).
        InvalidParameterError: Se a configuração não fizer sentido para estes dados.
    """
    frame = table.frame
    lineage = Lineage().then(
        LineageStep(
            stage=Stage.INPUT,
            operation="load",
            description=(
                f"Carregado {table.name}: {format_quantity(len(frame), 'linha', 'linhas')} × "
                f"{format_quantity(frame.shape[1], 'coluna', 'colunas')}."
            ),
            params=table.spec.to_dict(),
            rows_after=len(frame),
        )
    )
    cleaned = apply_cleaning(frame, config.cleaning, lineage)
    frame, lineage = cleaned.frame, cleaned.lineage
    notices: list[Notice] = list(cleaned.notices)
    if frame.empty:
        raise DataError("Depois da limpeza não sobrou nenhuma linha: revê os passos da limpeza.")

    task = suggest_task(frame, config.target)
    task_type = _choose_task(task, config)
    notices.extend(task.notices)
    frame, lineage = _drop_rows_without_target(frame, config.target, lineage)

    inferences = infer_types(frame)
    schema, schema_notices = _model_schema(inferences, config.target)
    notices.extend(schema_notices)
    features = schema.feature_names()
    if not features:
        raise DataError(
            "Não sobrou nenhuma feature que o modelo possa usar: confirma o alvo e as colunas "
            "removidas na limpeza."
        )
    lineage = lineage.then(_task_step(task, task_type, config, features, schema))

    classification = task_type.is_classification
    if config.split.strategy is SplitStrategy.STRATIFIED and not classification:
        raise InvalidParameterError(
            "A divisão estratificada só serve para classificação: escolhe a divisão aleatória."
        )
    indices, split_step = split_rows(
        frame,
        config.split,
        seed=seeds.seed_for(SPLIT_SEED),
        target=config.target if classification else None,
        root_seed=seeds.seed,
    )
    lineage = lineage.then(split_step)
    partitions = [part for part in Partition if part is Partition.TRAIN or len(indices.of(part))]
    parts = {part: frame.iloc[indices.of(part)] for part in partitions}

    pipeline = build_pipeline(schema, config.transforms)
    notices.extend(pipeline.notices)
    transformed = {Partition.TRAIN: pipeline.fit_transform(parts[Partition.TRAIN][list(features)])}
    for part in partitions[1:]:
        raw = parts[part][list(features)]
        transformed[part] = pipeline.transform(raw)
        notices.extend(_unknown_category_notices(pipeline.unknown_categories(raw), part))
    lineage = lineage.then(*pipeline.lineage_steps())

    feature_names = tuple(str(name) for name in transformed[Partition.TRAIN].columns)
    matrices = {part: to_matrix(transformed[part], part) for part in partitions}
    projection, projection_step = project_2d(matrices, feature_names, config.projection)
    if projection_step is not None:
        lineage = lineage.then(projection_step)

    labels = class_labels(frame[config.target]) if config.target and classification else ()
    splits = {
        part: DataSplit(
            X=matrices[part],
            y=target_values(parts[part][config.target], labels) if config.target else None,
            rows=parts[part].index.to_numpy(dtype="intp"),
        )
        for part in partitions
    }
    lineage = lineage.then(_output_step(splits, feature_names, labels))
    dataset = Dataset(
        name=table.name,
        task_type=task_type,
        feature_names=feature_names,
        target_name=config.target,
        class_labels=labels,
        splits=splits,
        schema=schema,
        lineage=lineage,
        projection=projection,
    )
    return PreparedData(dataset, frame, transformed, task, tuple(notices))


def _choose_task(task: TaskSuggestion, config: PreparationConfig) -> TaskType:
    errors = [notice.message for notice in task.notices if notice.level is NoticeLevel.ERROR]
    if errors or task.task_type is None:
        raise DataError(" ".join(errors) or task.reason)
    chosen = config.task_type or task.task_type
    if chosen not in task.allowed:
        allowed = ", ".join(option.label.lower() for option in task.allowed)
        if config.target is None:
            raise InvalidParameterError(
                f"{chosen.label} precisa de uma coluna-alvo. Sem alvo, as tarefas possíveis são: "
                f"{allowed}."
            )
        raise InvalidParameterError(
            f"{chosen.label} não serve para o alvo {config.target}. Escolhe uma destas: {allowed}."
        )
    return chosen


def _drop_rows_without_target(
    frame: pd.DataFrame, target: str | None, lineage: Lineage
) -> tuple[pd.DataFrame, Lineage]:
    if target is None or not frame[target].isna().any():
        return frame, lineage
    kept = frame.dropna(subset=[target])
    removed = len(frame) - len(kept)
    step = LineageStep(
        stage=Stage.TASK,
        operation="drop_missing_target",
        description=(
            f"{'Removida' if removed == 1 else 'Removidas'} "
            f"{format_quantity(removed, 'linha', 'linhas')} sem valor no alvo {target}."
        ),
        params={"target": target},
        rows_before=len(frame),
        rows_after=len(kept),
    )
    return kept, lineage.then(step)


def _model_schema(
    inferences: dict[str, TypeInference], target: str | None
) -> tuple[TableSchema, list[Notice]]:
    """O esquema para o modelo: texto livre, identificadores e texto por converter ficam de fora."""
    schema = schema_from(inferences, target)
    notices: list[Notice] = []
    as_text = [
        name
        for name, inference in inferences.items()
        if inference.stored_as_text and name != target and schema[name].role is ColumnRole.FEATURE
    ]
    if as_text:
        schema = schema.with_roles(dict.fromkeys(as_text, ColumnRole.IGNORED))
        notices.append(
            Notice.warning(
                "unconverted_text",
                f"{', '.join(as_text)} ficaram de fora do modelo porque estão guardadas como "
                "texto: converte-as na limpeza (operação convert_type) para as usar.",
            )
        )
    ignored = [c.name for c in schema if c.role is ColumnRole.IGNORED and c.name not in as_text]
    if ignored:
        notices.append(
            Notice.info(
                "ignored_columns",
                f"{', '.join(ignored)} não entram no modelo (texto livre ou identificadores). "
                "Podes removê-las na limpeza.",
            )
        )
    return schema, notices


def _task_step(
    task: TaskSuggestion,
    task_type: TaskType,
    config: PreparationConfig,
    features: tuple[str, ...],
    schema: TableSchema,
) -> LineageStep:
    origin = "sugerida" if config.task_type in (None, task.task_type) else "escolhida por ti"
    target = f"Alvo {config.target}" if config.target else "Sem alvo"
    description = (
        f"{target}; tarefa: {task_type.label.lower()} ({origin}). "
        f"{format_quantity(len(features), 'feature', 'features')}: {', '.join(features)}."
    )
    ignored = [column.name for column in schema if column.role is ColumnRole.IGNORED]
    if ignored:
        description += f" Ficam de fora: {', '.join(ignored)}."
    return LineageStep(
        stage=Stage.TASK,
        operation="choose_task",
        description=description,
        params={"target": config.target, "task_type": task_type.value},
    )


def _unknown_category_notices(counts: dict[str, int], partition: Partition) -> list[Notice]:
    notices: list[Notice] = []
    for column, count in counts.items():
        if count == 1:
            verbs = "não existia no treino e ficou"
        else:
            verbs = "não existiam no treino e ficaram"
        notices.append(
            Notice.info(
                "unknown_category",
                f"No {partition.label.lower()}, {format_quantity(count, 'valor', 'valores')} de "
                f"{column} {verbs} com zeros na codificação.",
                column,
            )
        )
    return notices


def _output_step(
    splits: dict[Partition, DataSplit], feature_names: tuple[str, ...], labels: tuple[str, ...]
) -> LineageStep:
    shapes = ", ".join(
        f"{part.label.lower()} {format_count(len(split))} × {len(feature_names)}"
        for part, split in splits.items()
    )
    description = f"Entregue em NumPy (float64): {shapes}."
    if labels:
        mapping = ", ".join(f"«{label}» → {index}" for index, label in enumerate(labels))
        description += (
            f" O alvo tem {format_quantity(len(labels), 'classe', 'classes')}, trocadas por "
            f"números: {mapping}."
        )
    return LineageStep(
        stage=Stage.OUTPUT,
        operation="to_numpy",
        description=description,
        params={"n_features": len(feature_names)},
        rows_after=len(splits[Partition.TRAIN]),
    )
