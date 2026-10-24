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

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_lab.core.data.boundary import class_labels, encode_classes, target_values, to_matrix
from ml_lab.core.data.cleaning.base import CleaningStep, apply_cleaning
from ml_lab.core.data.dataset import Dataset, DataSplit
from ml_lab.core.data.io.base import LoadedTable
from ml_lab.core.data.lineage import Lineage, LineageStep, Stage
from ml_lab.core.data.profiling.type_inference import TypeInference, infer_types, schema_from
from ml_lab.core.data.projection.projection_2d import ProjectionConfig, project_2d
from ml_lab.core.data.schema import ColumnRole, TableSchema
from ml_lab.core.data.splitting.base import (
    SplitConfig,
    SplitIndices,
    SplitStrategy,
    split_rows,
)
from ml_lab.core.data.task.task_detection import TaskSuggestion, suggest_task
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig, build_pipeline
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.formatting import format_count, format_quantity

SPLIT_SEED: Final = "split"
"""Nome do componente da divisão: a sua seed é ``seeds.seed_for("split")`` (ADR 0006)."""

FOLDS_SEED: Final = "folds"
"""Nome do componente dos folds da validação cruzada."""


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
    selected = _select(table, config)
    indices, split_step = _split(selected, config, seeds)
    partitions = [part for part in Partition if part is Partition.TRAIN or len(indices.of(part))]
    parts = {part: selected.frame.iloc[indices.of(part)] for part in partitions}
    dataset, transformed, notices = _assemble(
        table.name, selected, parts, config, selected.lineage.then(split_step)
    )
    return PreparedData(
        dataset, selected.frame, transformed, selected.task, (*selected.notices, *notices)
    )


def prepare_folds(
    table: LoadedTable, config: PreparationConfig, *, seeds: SeedManager, k: int
) -> tuple[Dataset, ...]:
    """Prepara os dados para a validação cruzada: um ``Dataset`` por fold.

    Divide os dados como ``prepare`` (com a mesma seed, por isso o teste é o mesmo e fica
    de fora) e parte o treino em ``k`` folds. No fold ``i``, a avaliação usa a parte ``i``
    e o treino as outras. As transformações aprendem de novo em cada fold, só com o treino
    desse fold: a parte de fora nunca influencia a escala nem a imputação.

    Raises:
        InvalidParameterError: Se ``k`` for menor do que 2 ou maior do que o treino.
    """
    selected = _select(table, config)
    indices, split_step = _split(selected, config, seeds)
    train = np.asarray(indices.train)
    if not 2 <= k <= len(train):
        raise InvalidParameterError(
            f"A validação cruzada precisa de entre 2 e {len(train)} folds; pediste {k}."
        )
    target = _class_column(selected.task_type, config.target)
    folds = _fold_positions(selected.frame.iloc[train], k, seeds.seed_for(FOLDS_SEED), target)
    datasets = []
    for number, held_out in enumerate(folds, start=1):
        kept = np.setdiff1d(np.arange(len(train)), held_out)
        parts = {
            Partition.TRAIN: selected.frame.iloc[train[kept]],
            Partition.TEST: selected.frame.iloc[train[held_out]],
        }
        step = LineageStep(
            stage=Stage.SPLIT,
            operation="cross_validation_fold",
            description=(
                f"Validação cruzada, fold {number} de {k}: treino com "
                f"{format_quantity(len(kept), 'linha', 'linhas')} e avaliação com as "
                f"{format_quantity(len(held_out), 'linha', 'linhas')} do fold."
            ),
            params={"fold": number, "k": k},
            rows_after=len(kept),
        )
        lineage = selected.lineage.then(split_step, step)
        dataset, _, _ = _assemble(table.name, selected, parts, config, lineage)
        datasets.append(dataset)
    return tuple(datasets)


@dataclass(frozen=True, slots=True, eq=False)
class _Selected:
    """A tabela depois das etapas 4 e 5: limpa, com a tarefa escolhida e as features."""

    frame: pd.DataFrame
    lineage: Lineage
    notices: tuple[Notice, ...]
    task: TaskSuggestion
    task_type: TaskType
    schema: TableSchema
    features: tuple[str, ...]


def _select(table: LoadedTable, config: PreparationConfig) -> _Selected:
    """Etapas 4 (limpeza) e 5 (tarefa): o que não depende da divisão."""
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
    return _Selected(frame, lineage, tuple(notices), task, task_type, schema, features)


def _split(
    selected: _Selected, config: PreparationConfig, seeds: SeedManager
) -> tuple[SplitIndices, LineageStep]:
    """Etapa 6: a divisão em treino, validação e teste."""
    classes = _class_column(selected.task_type, config.target)
    if config.split.strategy is SplitStrategy.STRATIFIED and classes is None:
        raise InvalidParameterError(
            "A divisão estratificada precisa de classes (um alvo categórico ou uma coluna de "
            "referência): escolhe a divisão aleatória."
        )
    return split_rows(
        selected.frame,
        config.split,
        seed=seeds.seed_for(SPLIT_SEED),
        target=classes,
        root_seed=seeds.seed,
    )


def _class_column(task_type: TaskType, target: str | None) -> str | None:
    """A coluna com classes: o alvo, na classificação, ou a referência, nas tarefas sem alvo.

    A referência nunca entra no treino, mas pode estratificar a divisão: com anomalias
    raras, uma divisão aleatória pode deixar o teste sem nenhuma, e sem elas não há
    precisão nem recall para comparar.
    """
    has_classes = task_type.is_classification or not task_type.requires_target
    return target if has_classes else None


def _class_labels(selected: _Selected, column: str) -> tuple[str, ...]:
    """As classes da coluna, por ordem; na deteção de anomalias, a mais rara fica com o 1.

    Uma referência de duas classes diz quais são as anomalias: a mais rara, contada na
    tabela toda. Dar-lhe o número 1 aqui, uma só vez, faz com que "1 = anomalia" valha
    igual no treino, no teste e em todos os folds.
    """
    labels = class_labels(selected.frame[column])
    if selected.task_type is TaskType.ANOMALY_DETECTION and len(labels) == 2:
        counts = np.bincount(encode_classes(selected.frame[column], labels), minlength=2)
        if counts[0] < counts[1]:
            labels = (labels[1], labels[0])
    return labels


def _assemble(
    name: str,
    selected: _Selected,
    parts: dict[Partition, pd.DataFrame],
    config: PreparationConfig,
    lineage: Lineage,
) -> tuple[Dataset, dict[Partition, pd.DataFrame], list[Notice]]:
    """Etapas 7 (transformação), 8 (projeção) e 9 (saída), a partir das partes já divididas."""
    features, partitions = list(selected.features), list(parts)
    notices: list[Notice] = []
    pipeline = build_pipeline(selected.schema, config.transforms)
    notices.extend(pipeline.notices)
    transformed = {Partition.TRAIN: pipeline.fit_transform(parts[Partition.TRAIN][features])}
    for part in partitions[1:]:
        raw = parts[part][features]
        transformed[part] = pipeline.transform(raw)
        notices.extend(_unknown_category_notices(pipeline.unknown_categories(raw), part))
    lineage = lineage.then(*pipeline.lineage_steps())

    feature_names = tuple(str(column) for column in transformed[Partition.TRAIN].columns)
    matrices = {part: to_matrix(transformed[part], part) for part in partitions}
    projection, projection_step = project_2d(matrices, feature_names, config.projection)
    if projection_step is not None:
        lineage = lineage.then(projection_step)

    target = config.target
    # Nas tarefas sem alvo, uma coluna-alvo é só uma referência (categórica) para comparar.
    classes = _class_column(selected.task_type, target)
    labels = _class_labels(selected, classes) if classes else ()
    splits = {
        part: DataSplit(
            X=matrices[part],
            y=target_values(parts[part][target], labels) if target else None,
            rows=parts[part].index.to_numpy(dtype="intp"),
        )
        for part in partitions
    }
    lineage = lineage.then(_output_step(splits, feature_names, labels))
    dataset = Dataset(
        name=name,
        task_type=selected.task_type,
        feature_names=feature_names,
        target_name=target,
        class_labels=labels,
        splits=splits,
        schema=selected.schema,
        lineage=lineage,
        projection=projection,
    )
    return dataset, transformed, notices


def _fold_positions(
    frame: pd.DataFrame, k: int, seed: int, target: str | None
) -> list[NDArray[np.intp]]:
    """As posições (dentro de ``frame``) de cada fold; estratificadas pelo alvo, se for dado.

    Na estratificação, as linhas de cada classe são baralhadas e distribuídas pelos folds
    à vez, continuando de classe para classe: todos os folds ficam com proporções
    parecidas de cada classe e tamanhos que diferem no máximo numa linha.
    """
    rng = np.random.default_rng(seed)
    if target is None:
        return [np.sort(fold) for fold in np.array_split(rng.permutation(len(frame)), k)]
    folds: list[list[int]] = [[] for _ in range(k)]
    turn = 0
    values = frame[target].astype(str).to_numpy()
    for value in sorted(set(values.tolist())):
        for position in rng.permutation(np.flatnonzero(values == value)):
            folds[turn % k].append(int(position))
            turn += 1
    return [np.sort(np.array(fold, dtype=np.intp)) for fold in folds]


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
    if config.target and not task_type.requires_target:
        target = f"Referência {config.target} (não entra no treino; serve para comparar)"
    else:
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
