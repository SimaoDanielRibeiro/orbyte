"""A história do treino: o que aconteceu num run, contado em capítulos curtos (ADR 0047).

O ecrã Explicar abre com ela, antes das explicações das previsões. Cada capítulo responde a
uma pergunta, pela ordem em que se faz:

1. **com que dados aprendeu**: as linhas, as features, a divisão e a preparação;
2. **como aprendeu**: os passos do treino e a curva da loss (ou "de uma vez");
3. **o que aprendeu**: os pesos, as features em que mais se apoia, as componentes…;
4. **como se saiu** em linhas que nunca viu, contra o palpite mais simples e contra o próprio
   treino (é aqui que se vê o overfitting);
5. **onde se engana**: o erro mais comum, na matriz de confusão, ou o tamanho dos erros;
6. **o que fazer a seguir**: os avisos do run, que dizem o que mudar.

O texto é curto e simples, com os números que contam a negrito (``**…**``). Cada capítulo
traz, quando ajuda, os dados de um visual (a divisão das linhas, a curva da loss, umas barras
ou a matriz de confusão), e a interface só os desenha. Tudo se escreve a partir do resultado
registado do run, por isso a história também se conta num run aberto do histórico, sem o
modelo em memória.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from ml_lab.core.data.lineage import LineageStep, Stage
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitStrategy
from ml_lab.core.data.transforms.scaling import ScalingMethod
from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.diagnostics import LUCKY_TEST_GAP, OVERFITTING_GAP, memorised
from ml_lab.core.evaluation.report import ConfusionMatrix, EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import ComponentNotFoundError
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import MODELS, ModelInfo
from ml_lab.core.services.experiments import ExperimentResult
from ml_lab.core.training.recorder import StepRecord
from ml_lab.formatting import (
    format_count,
    format_list,
    format_number,
    format_percent,
    format_quantity,
    format_significant,
)

type ChapterKey = Literal["data", "learning", "learned", "result", "errors", "next"]
"""O capítulo, por um nome estável (a interface escolhe o ícone por ele)."""

type BarTone = Literal["baseline", "train", "validation", "test", "up", "down", "plain"]
"""O papel de uma barra, para a interface lhe dar a cor: a referência (o palpite mais simples),
uma parte dos dados, um peso que sobe ou desce a previsão, ou uma barra sem papel."""

MAX_CURVE_POINTS: Final = 240
"""Pontos da curva da loss, no máximo: um treino longo resume-se (o primeiro e o último ficam)."""

TOP_FEATURES: Final = 8
"""As features nas barras do que o modelo aprendeu, no máximo."""

NAMED_FEATURES: Final = 3
"""As features que o texto nomeia."""

PREPARATION_POINTS: Final = 5
"""Os passos da preparação contados um a um, no máximo (os outros resumem-se numa linha)."""

EARLY_SHARE: Final = 0.3
"""Se 90% da descida da loss se fez até esta fração do treino, a loss "desceu depressa"."""

LATE_SHARE: Final = 0.85
"""Se só se chegou a 90% da descida depois desta fração, a loss "ainda descia no fim"."""

BIG_ERRORS: Final = 1.4
"""A razão RMSE / MAE a partir da qual uns poucos erros grandes pesam mais do que os outros."""

_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“])")
"""O fim de uma frase: o espaço depois de um ponto, antes de uma maiúscula ou de um número."""

_STEP_NAMES: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "max_epochs": ("epoch", "epochs"),
        "max_iter": ("iteration", "iterations"),
        "n_estimators": ("stage", "stages"),
        "max_depth": ("level", "levels"),
    }
)
"""O nome de um passo do treino, pelo hiperparâmetro que o limita."""

_HOW_SPLIT: Final[Mapping[SplitStrategy, str]] = MappingProxyType(
    {
        SplitStrategy.RANDOM: "at random",
        SplitStrategy.STRATIFIED: "at random, keeping the share of each class in each part",
        SplitStrategy.TIME: "by time, with the most recent rows in the test",
    }
)

_ROLES: Final[Mapping[Partition, str]] = MappingProxyType(
    {
        Partition.TRAIN: "to learn from",
        Partition.VALIDATION: "to check it along the way",
        Partition.TEST: "kept apart, never seen while learning",
    }
)


@dataclass(frozen=True, slots=True)
class SplitPart:
    """Uma parte das linhas do run: quantas são e para que servem.

    Attributes:
        partition: A parte.
        rows: Quantas linhas tem.
        share: A fração das linhas todas (de 0 a 1).
        role: Para que serve, numa frase curta ("to learn from").
    """

    partition: Partition
    rows: int
    share: float
    role: str


@dataclass(frozen=True, slots=True)
class StoryCurve:
    """A loss ao longo do treino, para o capítulo "como aprendeu".

    Attributes:
        step_label: O nome de um passo, para o eixo ("Epoch", "Iteration"…).
        steps: O número de cada passo (um treino longo vem resumido).
        loss: A loss no treino, em cada passo.
        validation: A loss na validação em cada passo, se o modelo a mediu (senão, vazio).
        best_step: O passo com a menor loss na validação, se houver.
    """

    step_label: str
    steps: tuple[int, ...]
    loss: tuple[float, ...]
    validation: tuple[float | None, ...] = ()
    best_step: int | None = None


@dataclass(frozen=True, slots=True)
class StoryBar:
    """Uma barra de um visual da história.

    Attributes:
        label: O nome a mostrar.
        value: O valor (uma fração, se as barras forem percentagens).
        tone: O papel da barra (a interface escolhe a cor por ele).
    """

    label: str
    value: float
    tone: BarTone


@dataclass(frozen=True, slots=True)
class StoryBars:
    """Umas barras: as métricas comparadas, os pesos ou a importância das features.

    Attributes:
        value_label: O que os valores são, para o eixo.
        as_percent: Se os valores se mostram como percentagens.
        bars: As barras, pela ordem a mostrar.
    """

    value_label: str
    as_percent: bool
    bars: tuple[StoryBar, ...]


@dataclass(frozen=True, slots=True)
class StoryChapter:
    """Um capítulo da história: uma pergunta, a resposta em poucas frases e um visual.

    Attributes:
        key: O capítulo, por um nome estável.
        title: A pergunta a que responde, como título.
        text: A resposta, com os números que contam a negrito (``**…**``).
        points: Linhas curtas a seguir ao texto (por exemplo, os passos da preparação).
        split: A divisão das linhas, para desenhar como uma barra.
        curve: A loss ao longo do treino.
        bars: Umas barras (as métricas comparadas, os pesos, a importância…).
        matrix: Uma matriz de confusão (o que é real nas linhas).
        matrix_rows: O título das linhas da matriz.
        matrix_columns: O título das colunas da matriz.
        notices: Os avisos que dizem o que fazer (no último capítulo).
    """

    key: ChapterKey
    title: str
    text: str
    points: tuple[str, ...] = ()
    split: tuple[SplitPart, ...] = ()
    curve: StoryCurve | None = None
    bars: StoryBars | None = None
    matrix: ConfusionMatrix | None = None
    matrix_rows: str = ""
    matrix_columns: str = ""
    notices: tuple[Notice, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingStory:
    """O que aconteceu no treino de um run, do princípio ao fim.

    Attributes:
        run_id: O run.
        headline: O resumo, numa frase.
        chapters: Os capítulos, pela ordem em que se leem.
    """

    run_id: str
    headline: str
    chapters: tuple[StoryChapter, ...]


def training_story(result: ExperimentResult, lineage: Sequence[LineageStep] = ()) -> TrainingStory:
    """Conta o treino de ``result`` (ver o módulo).

    Args:
        result: O resultado do run.
        lineage: A linhagem dos dados, se o run ainda estiver em memória: a preparação conta-se
            com os números concretos de cada passo (sem ela, pelo que a configuração pediu).
    """
    info = _info(result.model_name)
    chapters = [
        _data(result, info, lineage),
        _learning(result, info),
        _learned(result),
        _performance(result),
    ]
    errors = _errors(result)
    if errors is not None:
        chapters.append(errors)
    chapters.append(_next(result))
    return TrainingStory(result.run_id, _headline(result), tuple(chapters))


# --- o resumo ----------------------------------------------------------------------------


def _headline(result: ExperimentResult) -> str:
    model = result.model_display_name
    target = result.config.preparation.target
    report = result.main_report
    unseen = result.main_partition is Partition.TEST
    where = (
        f"on {format_count(report.n_samples)} rows it had never seen"
        if unseen
        else "on the rows it trained on"
    )
    task = result.task_type
    if task.is_classification and target:
        accuracy = _metric(report, "accuracy")
        if accuracy is not None:
            return (
                f"{model} learned to predict {target} and got {accuracy.formatted} right {where}."
            )
    if task is TaskType.REGRESSION and target:
        r2 = _metric(report, "r2")
        if r2 is not None and r2.value is not None:
            if r2.value > 0:
                return (
                    f"{model} learned to predict {target}: {where}, it explains "
                    f"{format_percent(r2.value)} of how it varies."
                )
            return (
                f"{model} tried to predict {target}, but {where} it does no better than always "
                "guessing the mean."
            )
    if task is TaskType.CLUSTERING:
        groups = _metric(report, "n_clusters")
        if groups is not None and groups.value is not None:
            found = format_quantity(int(groups.value), "group", "groups")
            return f"{model} split the rows into {found}."
    if task is TaskType.ANOMALY_DETECTION:
        rate = _metric(report, "anomaly_rate")
        if rate is not None:
            rows = "test rows" if unseen else "rows it trained on"
            flagged = "none" if rate.value == 0 else rate.formatted
            return f"{model} flagged {flagged} of the {rows} as anomalies."
    if task is TaskType.DIMENSIONALITY_REDUCTION:
        features = format_quantity(len(result.feature_names), "feature", "features")
        dimensions = int(result.hyperparameters.get("n_components", 2))
        written = format_quantity(dimensions, "dimension", "dimensions")
        return f"{model} summed up {features} in {written}."
    return f"{model} trained on {result.dataset_name}."


# --- 1. com que dados ----------------------------------------------------------------------


def _data(
    result: ExperimentResult, info: ModelInfo | None, lineage: Sequence[LineageStep]
) -> StoryChapter:
    preparation = result.config.preparation
    target = preparation.target
    features = format_quantity(len(result.feature_names), "feature", "features")
    task = result.task_type.label.lower()
    if target and result.task_type.requires_target:
        opening = (
            f"**{result.dataset_name}**, {task}: the model learns to predict **{target}** from "
            f"{features}."
        )
    else:
        opening = (
            f"**{result.dataset_name}**, {task}, with {features} and no target: the model looks "
            "for structure on its own."
        )
        if target:
            opening += (
                f" **{target}** stays out of the training: it's only a reference, to compare "
                "with at the end."
            )
    rows = {
        partition: result.reports[partition].n_samples
        for partition in Partition
        if partition in result.reports
    }
    total = sum(rows.values())
    split = tuple(
        SplitPart(partition, count, count / total if total else 0.0, _ROLES[partition])
        for partition, count in rows.items()
    )
    train = rows.get(Partition.TRAIN, 0)
    if Partition.TEST in rows:
        how = _HOW_SPLIT.get(preparation.split.strategy, "at random")
        sentence = (
            f"The {format_count(total)} rows were split {how} (seed {result.seed}): "
            f"**{format_count(train)} to learn from** and **{format_count(rows[Partition.TEST])} "
            "for the test**, rows the model never sees while it trains."
        )
        if Partition.VALIDATION in rows:
            sentence += (
                f" Another **{format_count(rows[Partition.VALIDATION])}** were kept for "
                "validation, to check the model along the way."
            )
    else:
        reason = (
            " (it only places the points it trained on)"
            if info is not None and not info.supports_new_data
            else ""
        )
        sentence = (
            f"It trained on all **{format_count(train)}** rows{reason}, with no separate test."
        )
    return StoryChapter(
        "data",
        "What it learned from",
        f"{opening} {sentence}",
        points=_preparation(preparation, lineage),
        split=split,
    )


def _preparation(preparation: PreparationConfig, lineage: Sequence[LineageStep]) -> tuple[str, ...]:
    """A preparação, passo a passo: pela linhagem, se houver; senão, pela configuração."""
    if lineage:
        # Só a primeira frase de cada passo: o pormenor todo está no ecrã Preparar.
        steps = [
            _first_sentence(step.description)
            for step in lineage
            if step.stage in (Stage.CLEANING, Stage.TRANSFORM)
        ]
    else:
        steps = _configured(preparation)
    if not steps:
        return ("No cleaning and no transformations: the data went in as it was.",)
    if len(steps) > PREPARATION_POINTS:
        rest = len(steps) - (PREPARATION_POINTS - 1)
        more = format_quantity(rest, "more step", "more steps")
        steps = [*steps[: PREPARATION_POINTS - 1], f"…and {more}, in Prepare."]
    return tuple(steps)


def _configured(preparation: PreparationConfig) -> list[str]:
    """O que a configuração pediu (a linhagem, com os números, só existe com o run em memória)."""
    steps = []
    if preparation.cleaning:
        cleaning = format_quantity(len(preparation.cleaning), "cleaning step", "cleaning steps")
        steps.append(f"{cleaning.capitalize()} on the table.")
    transforms = preparation.transforms
    if transforms.scaling is ScalingMethod.STANDARD:
        steps.append(
            "Numbers standardised: mean 0 and standard deviation 1, from the training rows."
        )
    elif transforms.scaling is ScalingMethod.MINMAX:
        steps.append("Numbers rescaled between 0 and 1, with the limits of the training rows.")
    if transforms.polynomial_degree > 1:
        steps.append(f"Polynomial features up to degree {transforms.polynomial_degree}.")
    if transforms.binned_columns:
        columns = format_list(list(transforms.binned_columns))
        steps.append(f"{columns} split into {transforms.n_bins} bins.")
    return steps


# --- 2. como aprendeu ----------------------------------------------------------------------


def _learning(result: ExperimentResult, info: ModelInfo | None) -> StoryChapter:
    model = result.model_display_name
    about = f"**{model}**: {info.summary}" if info is not None else f"**{model}**."
    records = [
        record
        for record in result.history
        if record.loss is not None and math.isfinite(record.loss)
    ]
    took = _duration(result.duration_seconds)
    title = "How it learned"
    points = tuple(
        notice.message for notice in result.training_notices if notice.level is NoticeLevel.INFO
    )
    if len(records) < 2:
        text = f"{about} It learned in one go, in {took}, so there are no steps to watch."
        return StoryChapter("learning", title, text, points=points)
    limit = info.step_limit if info is not None else None
    singular, plural = _STEP_NAMES.get(limit or "", ("step", "steps"))
    steps = records[-1].step
    sentences = [
        about,
        f"It trained for **{format_quantity(steps, singular, plural)}**, in {took}.",
        _loss_story(records, result.task_type),
    ]
    curve = _curve(records, singular)
    if curve.best_step is not None:
        sentences.append(
            "On the rows it set aside for validation, which it doesn't learn from, the loss was "
            f"lowest at {singular} **{format_count(curve.best_step)}**."
        )
    if result.stopped_early:
        sentences.append(f"You stopped it at {singular} {format_count(steps)}.")
    return StoryChapter("learning", title, " ".join(sentences), points=points, curve=curve)


def _loss_story(records: Sequence[StepRecord], task: TaskType) -> str:
    """Como a loss mudou: quanto desceu, e se desceu depressa ou ainda descia no fim."""
    start, end = _loss(records[0]), _loss(records[-1])
    # A loss do clustering não é um erro (não há respostas certas): é o ajuste dos grupos.
    meaning = (
        "how poorly the groups fit the training rows"
        if task is TaskType.CLUSTERING
        else "how wrong it is on the training rows"
    )
    opening = (
        f"The loss, {meaning}, went from **{format_significant(start, 3)}** to "
        f"**{format_significant(end, 3)}**"
    )
    scale = max(abs(start), 1e-12)
    change = (end - start) / scale
    if change <= -0.01:
        drop = start - end
        lower = f", {format_percent(-change, 0)} lower" if start > 0 else ""
        # Onde se fez a maior parte da descida: logo no início, ou só perto do fim?
        reached = next(
            index for index, record in enumerate(records) if start - _loss(record) >= 0.9 * drop
        )
        when = reached / (len(records) - 1)
        if when <= EARLY_SHARE:
            return f"{opening}{lower}: most of the drop came early, and then it levelled off."
        if when >= LATE_SHARE:
            return f"{opening}{lower}: it was still falling at the end, so more steps might help."
        return f"{opening}{lower}, falling steadily."
    if change >= 0.01:
        return f"{opening}: it went up, so the training didn't work (the warnings say why)."
    return f"{opening}: it barely changed, because it started close to where it ended."


def _curve(records: Sequence[StepRecord], singular: str) -> StoryCurve:
    chosen = _thin(records, MAX_CURVE_POINTS)
    validation = tuple(record.metrics.get("val_loss") for record in chosen)
    measured = [
        (value, record.step)
        for record in records
        if (value := record.metrics.get("val_loss")) is not None and math.isfinite(value)
    ]
    return StoryCurve(
        step_label=singular.capitalize(),
        steps=tuple(record.step for record in chosen),
        loss=tuple(_loss(record) for record in chosen),
        validation=validation if measured else (),
        best_step=min(measured)[1] if measured else None,
    )


def _thin(records: Sequence[StepRecord], limit: int) -> list[StepRecord]:
    """No máximo ``limit`` passos, espalhados pelo treino (com o primeiro e o último)."""
    if len(records) <= limit:
        return list(records)
    last = len(records) - 1
    picked = sorted({round(index * last / (limit - 1)) for index in range(limit)})
    return [records[index] for index in picked]


def _loss(record: StepRecord) -> float:
    assert record.loss is not None  # só se chamam os passos com loss
    return float(record.loss)


# --- 3. o que aprendeu ---------------------------------------------------------------------


def _learned(result: ExperimentResult) -> StoryChapter:
    title = "What it learned"
    learned = result.learned
    if learned.weights is not None:
        return _weights(result)
    if learned.importances:
        return _importances(result)
    ratios = result.model_state.get("explained_variance_ratio")
    if isinstance(ratios, list) and ratios:
        return _variance(result, [float(ratio) for ratio in ratios])
    text = _from_properties(result)
    if text is None:
        text = " ".join(learned.notes) or (
            "What it learned doesn't fit in a few numbers: Train & evaluate shows it as it is."
        )
    return StoryChapter("learned", title, text)


def _weights(result: ExperimentResult) -> StoryChapter:
    weights = result.learned.weights
    assert weights is not None  # quem chama confirmou
    scaled = result.config.preparation.transforms.scaling is not None
    comparable = (
        "The features were scaled first, so the sizes of the weights can be compared."
        if scaled
        else "The features weren't scaled, so a big weight may only mean a feature with small "
        "numbers."
    )
    if len(weights.weights) == 1:
        values = weights.weights[0]
        subject = (
            f"the chance of “{weights.outputs[0]}”"
            if result.task_type.is_classification
            else "the prediction"
        )
        ranked = sorted(zip(weights.features, values, strict=True), key=lambda pair: -abs(pair[1]))
        named = [
            f"**{name}** {'pushes it up' if value > 0 else 'pushes it down'} ({_signed(value)})"
            for name, value in ranked[:NAMED_FEATURES]
        ]
        text = (
            "It gave each feature a weight, and adds them up to predict. The features that move "
            f"{subject} the most: {format_list(named)}. {comparable}"
        )
        bars = StoryBars(
            f"Weight on {subject}",
            False,
            tuple(
                StoryBar(name, value, "up" if value >= 0 else "down")
                for name, value in ranked[:TOP_FEATURES]
            ),
        )
        return StoryChapter("learned", "What it learned", text, bars=bars)
    # Várias classes: um peso por feature e por classe. O tamanho médio diz o que mais pesa.
    sizes = [
        sum(abs(row[column]) for row in weights.weights) / len(weights.weights)
        for column in range(len(weights.features))
    ]
    ranked = sorted(zip(weights.features, sizes, strict=True), key=lambda pair: -pair[1])
    named = [f"**{name}**" for name, _ in ranked[:NAMED_FEATURES]]
    text = (
        f"It learned one weight per feature for each of the {len(weights.outputs)} classes, and "
        f"picks the class with the highest sum. The features that weigh the most overall: "
        f"{format_list(named)}. {comparable}"
    )
    bars = StoryBars(
        "Mean size of the weights",
        False,
        tuple(StoryBar(name, size, "plain") for name, size in ranked[:TOP_FEATURES]),
    )
    return StoryChapter("learned", "What it learned", text, bars=bars)


def _importances(result: ExperimentResult) -> StoryChapter:
    learned = result.learned
    pairs = learned.importances[:TOP_FEATURES]
    named = [
        f"**{name}** ({format_percent(value, 0)})"
        for name, value in pairs[:NAMED_FEATURES]
        if value > 0
    ]
    sentences = [
        f"The features it relies on most are {format_list(named)}, as a share of all its "
        "importance: how much each one helped it split the rows."
        if named
        else "No feature helped it split the rows."
    ]
    root = learned.tree[0] if learned.tree else None
    if root is not None and root.feature is not None and root.threshold is not None:
        scaled = result.config.preparation.transforms.scaling is not None
        units = " (on the scaled values of the preparation)" if scaled else ""
        sentences.append(
            "Its first question, the one that splits the data best, is "
            f"“{root.feature} ≤ {format_number(root.threshold, 3)}?”{units}."
        )
    properties = {item.name: item.value for item in learned.properties}
    # Uma floresta junta árvores independentes, que votam; o boosting junta-as em sequência.
    stages, trees = properties.get("n_stages"), properties.get("n_trees")
    if isinstance(stages, int) and stages > 1:
        sentences.append(
            f"It's made of **{format_count(stages)} small trees**, each one correcting the "
            "ones before it."
        )
    elif isinstance(trees, int) and trees > 1:
        sentences.append(f"It's made of **{format_count(trees)} trees** that vote together.")
    bars = StoryBars(
        "Share of the importance",
        True,
        tuple(StoryBar(name, value, "plain") for name, value in pairs),
    )
    return StoryChapter("learned", "What it learned", " ".join(sentences), bars=bars)


def _variance(result: ExperimentResult, ratios: list[float]) -> StoryChapter:
    features = format_quantity(len(result.feature_names), "feature", "features")
    components = format_quantity(len(ratios), "component", "components")
    text = (
        "It found new directions in the data, the components, ordered by how much of the "
        f"variation each one keeps. The {components} keep **{format_percent(sum(ratios))}** of "
        f"the variation of the {features}; the first alone keeps "
        f"**{format_percent(ratios[0])}**."
    )
    bars = StoryBars(
        "Share of the variance",
        True,
        tuple(
            StoryBar(f"Component {number}", ratio, "plain")
            for number, ratio in enumerate(ratios, start=1)
        ),
    )
    return StoryChapter("learned", "What it learned", text, bars=bars)


def _from_properties(result: ExperimentResult) -> str | None:
    """O que aprendeu, pelos números simples do estado (os vizinhos, a rede, os grupos…)."""
    properties = {item.name: item.value for item in result.learned.properties}
    kept = properties.get("n_train")
    neighbours = result.hyperparameters.get("n_neighbors")
    if isinstance(kept, int) and isinstance(neighbours, int):
        return (
            f"It doesn't sum the data up: it keeps all **{format_count(kept)} training rows** "
            f"and, for each new point, lets the **{neighbours}** closest vote."
        )
    support = properties.get("n_support")
    if isinstance(support, str):  # por classe, "12 + 15 = 27": conta o total
        support = support.rpartition("=")[2].strip()
    if support is not None:
        return (
            f"Only **{support}** training rows ended up mattering: the support vectors, the "
            "points closest to the boundary."
        )
    architecture = properties.get("architecture")
    weights = properties.get("n_parameters")
    if isinstance(architecture, str) and isinstance(weights, int):
        activation = properties.get("activation")
        each = f" Each hidden neuron uses {activation}." if isinstance(activation, str) else ""
        return (
            f"It's a network of **{architecture}** neurons, layer by layer, with "
            f"**{format_count(weights)}** weights learned together.{each}"
        )
    groups, noise = properties.get("n_clusters"), properties.get("n_noise")
    if isinstance(groups, int) and isinstance(noise, int):
        return (
            f"It found **{format_quantity(groups, 'dense group', 'dense groups')}** and left "
            f"**{format_count(noise)}** points as noise, in no group."
        )
    if any(table.title.startswith("Centres") for table in result.learned.tables):
        return (
            "Each group is summed up by its centre, the average point of the group: a new "
            "point goes to the closest centre."
        )
    if any(table.title.startswith("Means of each class") for table in result.learned.tables):
        return (
            "It learned, for each class, the mean and the spread of each feature, and picks the "
            "class that makes a row the most likely."
        )
    threshold = properties.get("threshold")
    if isinstance(threshold, float):
        return (
            "It gives every point an anomaly score and flags those above the threshold, "
            f"**{format_significant(threshold, 3)}**."
        )
    divergence = properties.get("kl_divergence")
    if isinstance(divergence, float):
        return (
            "It placed the points on a 2D map, keeping neighbours close together (the final KL "
            f"divergence, lower is better, is **{format_significant(divergence, 3)}**)."
        )
    return None


# --- 4. como se saiu -----------------------------------------------------------------------


def _performance(result: ExperimentResult) -> StoryChapter:
    title = "How well it did"
    task = result.task_type
    name = PRIMARY_METRIC.get(task)
    main = result.main_report
    metric = _metric(main, name) if name is not None else None
    if metric is None or metric.value is None:
        return StoryChapter("result", title, "This run has no main metric to sum it up.")
    rows = (
        f"**{format_count(main.n_samples)} test rows**, which it never saw"
        if result.main_partition is Partition.TEST
        else f"**{format_count(main.n_samples)} rows** it trained on"
    )
    train = _metric(result.reports.get(Partition.TRAIN), metric.name)
    gap = _gap(train, metric, main.n_samples) if result.main_partition is Partition.TEST else ""
    if task.is_classification:
        return _classification(result, metric, rows, gap)
    if task is TaskType.REGRESSION:
        return _regression(result, metric, rows, gap)
    if task is TaskType.CLUSTERING:
        return _clustering(result, metric, gap)
    if task is TaskType.ANOMALY_DETECTION:
        return _anomalies(result, metric, rows)
    if task is TaskType.DIMENSIONALITY_REDUCTION:
        perfect = "100%" if metric.as_percent else "1"
        text = (
            f"On the {rows}, its trustworthiness is **{metric.formatted}**: how often the points "
            "that are neighbours on the map were also neighbours in the data, where "
            f"{perfect} is perfect."
        )
    else:
        text = f"On the {rows}, its {metric.display_name} is **{metric.formatted}**."
    return StoryChapter("result", title, text, bars=_compared(result, metric))


def _classification(
    result: ExperimentResult, metric: MetricResult, rows: str, gap: str
) -> StoryChapter:
    assert metric.value is not None  # quem chama confirmou
    sentences = [f"On the {rows}, it got **{metric.formatted}** right: that's its accuracy."]
    extra: list[StoryBar] = []
    majority = _majority(result.main_report.confusion_matrix)
    if majority is not None:
        label, share = majority
        difference = metric.value - share
        if difference > 0.02:
            verdict = f"the model is **{_points(difference)} points better**"
        elif difference >= -0.02:
            verdict = "the model is **no better** than that"
        else:
            verdict = f"the model is **{_points(-difference)} points worse**"
        sentences.append(
            f"Always guessing “{label}”, the most common class, would get "
            f"**{format_percent(share)}**: {verdict}."
        )
        extra.append(StoryBar(f"Always “{label}”", share, "baseline"))
    if gap:
        sentences.append(gap)
    return StoryChapter(
        "result",
        "How well it did",
        " ".join(sentences),
        bars=_compared(result, metric, extra),
    )


def _regression(
    result: ExperimentResult, metric: MetricResult, rows: str, gap: str
) -> StoryChapter:
    assert metric.value is not None  # quem chama confirmou
    target = result.config.preparation.target or "the target"
    if metric.value > 0:
        reading = (
            f"it explains **{format_percent(metric.value)}** of how {target} varies, where "
            "always guessing the mean explains none"
        )
    else:
        reading = "no better than always guessing the mean, which scores 0"
    sentences = [f"On the {rows}, R² is **{metric.formatted}**: {reading}."]
    mae = _metric(result.main_report, "mae")
    if mae is not None:
        sentences.append(f"On average, it misses by **{_amount(mae)}** (MAE).")
    if gap:
        sentences.append(gap)
    baseline = [StoryBar("Always the mean", 0.0, "baseline")]
    return StoryChapter(
        "result",
        "How well it did",
        " ".join(sentences),
        bars=_compared(result, metric, baseline),
    )


def _clustering(result: ExperimentResult, metric: MetricResult, gap: str) -> StoryChapter:
    assert metric.value is not None  # quem chama confirmou
    main = result.main_report
    groups = _metric(main, "n_clusters")
    found = (
        f"It found **{format_quantity(int(groups.value), 'group', 'groups')}**. "
        if groups is not None and groups.value is not None
        else ""
    )
    if metric.value >= 0.5:
        reading = "the groups are well apart"
    elif metric.value >= 0.25:
        reading = "the groups are there, but they touch"
    else:
        reading = "the groups overlap a lot, or there's little structure to find"
    sentences = [f"{found}The silhouette is **{metric.formatted}**, from −1 to 1: {reading}."]
    noise = _metric(main, "noise_ratio")
    if noise is not None and noise.value is not None and noise.value > 0:
        sentences.append(f"**{noise.formatted}** of the points were left as noise.")
    agreement = _metric(main, "adjusted_rand")
    if agreement is not None:
        sentences.append(
            f"Against the reference column, the agreement (adjusted Rand) is "
            f"**{agreement.formatted}**: 1 is a perfect match, and 0 is what chance gives."
        )
    if gap:
        sentences.append(gap)
    return StoryChapter(
        "result", "How well it did", " ".join(sentences), bars=_compared(result, metric)
    )


def _anomalies(result: ExperimentResult, metric: MetricResult, rows: str) -> StoryChapter:
    assert metric.value is not None  # quem chama confirmou
    main = result.main_report
    flagged = "none" if metric.value == 0 else metric.formatted
    sentences = [f"Of the {rows}, it flagged **{flagged}** as anomalies."]
    recall = _metric(main, "anomaly_recall")
    precision = _metric(main, "anomaly_precision")
    part = "test" if main.partition is Partition.TEST else "training rows"
    bars = None
    if recall is not None and precision is not None:
        assert recall.value is not None  # _metric confirma
        assert precision.value is not None
        sentences.append(
            f"Against the reference column, it caught **{recall.formatted}** of the real "
            f"anomalies, and **{precision.formatted}** of what it flagged was real."
        )
        bars = StoryBars(
            f"On the {part}",
            True,
            (
                StoryBar("Real anomalies caught (recall)", recall.value, "test"),
                StoryBar("Flags that were real (precision)", precision.value, "test"),
            ),
        )
    elif recall is not None:  # não marcou nada: não há precisão, e o recall é 0
        sentences.append("So it caught **none** of the real anomalies in the reference column.")
        real = _anomaly_share(main.confusion_matrix)
        if real is not None:
            bars = StoryBars(
                f"Share of the {'test rows' if part == 'test' else part}",
                True,
                (
                    StoryBar("Flagged as anomalies", metric.value, "test"),
                    StoryBar("Real anomalies (the reference)", real, "baseline"),
                ),
            )
    return StoryChapter("result", "How well it did", " ".join(sentences), bars=bars)


def _anomaly_share(matrix: ConfusionMatrix | None) -> float | None:
    """A fração de anomalias verdadeiras, pela matriz da referência (a segunda linha)."""
    if matrix is None or len(matrix.counts) != 2:
        return None
    total = sum(sum(row) for row in matrix.counts)
    return sum(matrix.counts[1]) / total if total else None


def _compared(
    result: ExperimentResult, metric: MetricResult, first: Sequence[StoryBar] = ()
) -> StoryBars | None:
    """A métrica em cada parte dos dados (depois das barras de ``first``, a referência)."""
    tones: Mapping[Partition, BarTone] = {
        Partition.TRAIN: "train",
        Partition.VALIDATION: "validation",
        Partition.TEST: "test",
    }
    bars = [*first]
    for partition in Partition:
        found = _metric(result.reports.get(partition), metric.name)
        if found is not None and found.value is not None:
            bars.append(StoryBar(partition.label, found.value, tones[partition]))
    if len(bars) < 2:
        return None
    return StoryBars(metric.display_name, metric.as_percent, tuple(bars))


def _gap(train: MetricResult | None, test: MetricResult, n_test: int) -> str:
    """O que a diferença entre o treino e o teste diz (o overfitting vê-se aqui)."""
    if train is None or train.value is None or test.value is None:
        return ""
    if train.higher_is_better is None:  # a métrica só descreve: não há melhor nem pior
        return ""
    gap = train.value - test.value
    if not train.higher_is_better:
        gap = -gap
    difference = f"{_points(abs(gap))} points" if train.as_percent else format_number(abs(gap), 3)
    # A mesma regra do aviso do treino (`memorised`): a história e o aviso nunca se contradizem.
    overfits = (
        memorised(train.value, test.value, n_test)
        if train.higher_is_better
        else gap >= OVERFITTING_GAP
    )
    if overfits:
        return (
            f"On the training rows it scored **{train.formatted}**, {difference} better: it "
            "**memorised** part of the training data instead of the general pattern "
            "(overfitting)."
        )
    if gap <= -LUCKY_TEST_GAP:
        return (
            f"On the training rows it scored {train.formatted}, worse than on the test: with a "
            "small test, that can be luck."
        )
    if gap > 0.03:
        return (
            f"On the training rows it scored {train.formatted}, a little better, which is "
            "normal: a model always knows its own examples best."
        )
    return (
        f"On the training rows it scored **{train.formatted}**, almost the same: what it learned "
        "**holds on new rows**."
    )


# --- 5. onde se engana ---------------------------------------------------------------------


def _errors(result: ExperimentResult) -> StoryChapter | None:
    title = "Where it goes wrong"
    main = result.main_report
    matrix = main.confusion_matrix
    task = result.task_type
    if matrix is not None and task.is_classification:
        return StoryChapter(
            "errors",
            title,
            _mistakes(matrix),
            matrix=matrix,
            matrix_rows="Actual class",
            matrix_columns="Predicted class",
        )
    if matrix is not None and task is TaskType.ANOMALY_DETECTION and len(matrix.counts) == 2:
        missed, alarms = matrix.counts[1][0], matrix.counts[0][1]
        text = (
            f"Against the reference column, it missed **{format_count(missed)}** real "
            f"{'anomaly' if missed == 1 else 'anomalies'} and raised "
            f"**{format_count(alarms)}** false {'alarm' if alarms == 1 else 'alarms'}, normal "
            "rows it flagged."
        )
        return StoryChapter(
            "errors",
            title,
            text,
            matrix=matrix,
            matrix_rows="Reference",
            matrix_columns="Flagged as",
        )
    if matrix is not None and task is TaskType.CLUSTERING:
        text = (
            "Against the reference column: each row of the table is a real class, and each "
            "column a group the model found. A good grouping puts each class mostly in one group."
        )
        return StoryChapter(
            "errors",
            title,
            text,
            matrix=matrix,
            matrix_rows="Reference class",
            matrix_columns="Group",
        )
    if task is TaskType.REGRESSION:
        return _error_sizes(main)
    return None


def _mistakes(matrix: ConfusionMatrix) -> str:
    """O erro mais comum de uma matriz de confusão, em palavras."""
    counts = matrix.counts
    if not counts or not counts[0]:
        return "There were no rows to compare."
    total = sum(sum(row) for row in counts)
    right = sum(counts[index][index] for index in range(min(len(counts), len(counts[0]))))
    wrong = total - right
    if wrong == 0:
        return f"It got all **{format_count(total)}** rows right: there's no mistake to show."
    count, actual, predicted = max(
        (counts[row][column], row, column)
        for row in range(len(counts))
        for column in range(len(counts[row]))
        if row != column
    )
    columns = matrix.columns
    return (
        f"It got **{format_count(wrong)} of the {format_count(total)}** rows wrong. Its most "
        f"common mistake: predicting “{columns[predicted]}” for rows that were "
        f"“{matrix.labels[actual]}” (**{format_quantity(count, 'time', 'times')}**). The diagonal "
        "of the table is what it got right."
    )


def _error_sizes(main: EvaluationReport) -> StoryChapter | None:
    mae, rmse = _metric(main, "mae"), _metric(main, "rmse")
    if mae is None or rmse is None or mae.value is None or rmse.value is None or mae.value <= 0:
        return None
    if rmse.value / mae.value >= BIG_ERRORS:
        reading = (
            "a few big misses weigh more than the typical error: the residuals in Train & "
            "evaluate show where they are."
        )
    else:
        reading = "the errors are fairly even in size, with no huge misses."
    text = (
        f"On average it misses by **{_amount(mae)}** (MAE), and the RMSE, which punishes big "
        f"errors more, is **{_amount(rmse)}**: {reading}"
    )
    bars = StoryBars(
        "Error on the test",
        False,
        (
            StoryBar("MAE (the typical error)", mae.value, "test"),
            StoryBar("RMSE (big errors weigh more)", rmse.value, "test"),
        ),
    )
    return StoryChapter("errors", "Where it goes wrong", text, bars=bars)


# --- 6. o que fazer a seguir ---------------------------------------------------------------


def _next(result: ExperimentResult) -> StoryChapter:
    title = "What to try next"
    seen: set[tuple[str, str]] = set()
    warnings = []
    for notice in (*result.notices, *result.main_report.notices):
        key = (notice.code, notice.message)
        if notice.level is NoticeLevel.WARNING and key not in seen:
            seen.add(key)
            warnings.append(notice)
    if warnings:
        things = "One thing" if len(warnings) == 1 else f"{format_count(len(warnings))} things"
        text = f"{things} to look at before you trust this model:"
        return StoryChapter("next", title, text, notices=tuple(warnings))
    text = (
        "Nothing went wrong in this run: there are no warnings. Next, save the model to use it "
        "elsewhere, or train another one and compare the two in History."
    )
    return StoryChapter("next", title, text)


# --- ajudas --------------------------------------------------------------------------------


def _info(name: str) -> ModelInfo | None:
    try:
        return MODELS.get(name).info
    except ComponentNotFoundError:  # um run antigo, de um modelo que já não existe
        return None


def _metric(report: EvaluationReport | None, name: str) -> MetricResult | None:
    """Uma métrica de um relatório, se existir e tiver valor."""
    if report is None:
        return None
    found = report.find(name)
    if found is None or found.value is None or math.isnan(found.value):
        return None
    return found


def _majority(matrix: ConfusionMatrix | None) -> tuple[str, float] | None:
    """A classe mais comum nas linhas avaliadas, e a parte delas que é dessa classe."""
    if matrix is None:
        return None
    totals = [sum(row) for row in matrix.counts]
    rows = sum(totals)
    if rows == 0:
        return None
    largest = max(range(len(totals)), key=lambda index: totals[index])
    return matrix.labels[largest], totals[largest] / rows


def _first_sentence(text: str) -> str:
    """A primeira frase de ``text`` (acaba num ponto seguido de uma maiúscula ou de um número)."""
    return _SENTENCE_END.split(text, maxsplit=1)[0]


def _points(fraction: float) -> str:
    """Uma diferença entre duas percentagens, em pontos percentuais."""
    return format_number(fraction * 100, 1)


def _signed(value: float) -> str:
    """Um peso com o sinal à frente (``+0.41``, ``−2.51``, ``+72,914``)."""
    written = format_significant(abs(value), 3)
    return f"+{written}" if value >= 0 else f"−{written}"


def _amount(metric: MetricResult) -> str:
    """Um erro na escala do alvo, sem algarismos a mais (``50,356`` e não ``50,355.920``)."""
    assert metric.value is not None  # quem chama confirmou
    return format_significant(metric.value, 4)


def _duration(seconds: float) -> str:
    """Quanto tempo demorou, escrito para pessoas."""
    if seconds < 1:
        return f"{format_number(seconds, 2)} s"
    if seconds < 60:
        return f"{format_number(seconds, 1)} s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s"
