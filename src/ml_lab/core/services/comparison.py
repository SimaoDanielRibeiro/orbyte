"""Comparar runs lado a lado: as métricas, as escolhas que mudam e as curvas da loss.

Uma comparação só ensina alguma coisa se os runs diferirem numa escolha de cada vez. Por
isso, além das métricas de cada run, mostra só as escolhas que mudam entre eles (o
modelo, um hiperparâmetro, a seed, um passo da preparação), e lê o resultado:

- mesma configuração e mesma seed têm de dar as mesmas métricas (reprodutibilidade);
- só a seed muda: a diferença é o efeito do acaso;
- mudam várias escolhas ao mesmo tempo: não se sabe qual fez a diferença.

Serve os runs de modelos e os de agentes (reinforcement learning): num agente, as escolhas
são o ambiente, os seus parâmetros, o agente e os seus hiperparâmetros.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from ml_lab.core.data.cleaning.base import CLEANING, CleaningStep
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.report import EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import ComponentNotFoundError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition
from ml_lab.core.models.base import MODELS
from ml_lab.core.rl.agents.base import AGENTS
from ml_lab.core.rl.environments.base import ENVIRONMENTS
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.services.experiments import ExperimentResult
from ml_lab.core.services.reinforcement import ReinforcementResult
from ml_lab.core.tracking.store import RunEntry
from ml_lab.formatting import format_number, format_percent, format_significant

type Compared = ExperimentResult | ReinforcementResult

MAX_COMPARED: Final = 5
"""Quantos runs se comparam, no máximo (mais do que isto já não se lê lado a lado)."""

LOSS_POINTS: Final = 300
"""Pontos de cada curva da loss, no máximo."""

_PREPARATION_FIELDS: Final[Mapping[str, str]] = {
    # A divisão.
    "strategy": "strategy",
    "test_size": "test share",
    "validation_size": "validation share",
    "time_column": "time column",
    # As transformações.
    "numeric_imputation": "numeric nulls",
    "categorical_imputation": "categorical nulls",
    "scaling": "scaling",
    "date_features": "date features",
    "polynomial_degree": "polynomial degree",
    "binned_columns": "binned columns",
    "n_bins": "number of bins",
    # A projeção 2D.
    "method": "method",
    "columns": "columns",
}
"""O nome a mostrar de cada escolha da preparação (o resto do nome é a parte a que pertence)."""

SEED_CHOICE: Final = "Seed"
MODEL_CHOICE: Final = "Model"
AGENT_CHOICE: Final = "Agent"
"""Os nomes das escolhas que a leitura da comparação procura (só a seed mudou? o modelo?)."""


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Uma métrica em cada run, na mesma parte dos dados.

    Attributes:
        name: O identificador da métrica.
        display_name: O nome a mostrar.
        as_percent: Se se mostra como percentagem.
        higher_is_better: Se maior é melhor (``None`` nas que só descrevem).
        values: O valor em cada run, pela ordem dos runs (``None`` se não tiver).
        best: A posição do melhor run, se houver um só melhor.
    """

    name: str
    display_name: str
    as_percent: bool
    higher_is_better: bool | None
    values: tuple[float | None, ...]
    best: int | None = None


class ChoiceGroup(StrEnum):
    """A parte do run a que uma escolha pertence (a comparação arruma as diferenças por elas).

    A ordem é a de leitura: primeiro o que define o run (os dados, a tarefa, o modelo, a seed),
    depois a preparação dos dados, o ambiente (no reinforcement learning) e os hiperparâmetros.
    """

    RUN = "run"
    PREPARATION = "preparation"
    ENVIRONMENT = "environment"
    HYPERPARAMETERS = "hyperparameters"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface."""
        return _CHOICE_GROUP_LABELS[self]


_CHOICE_GROUP_LABELS: Final[Mapping[ChoiceGroup, str]] = MappingProxyType(
    {
        ChoiceGroup.RUN: "The run",
        ChoiceGroup.PREPARATION: "The preparation",
        ChoiceGroup.ENVIRONMENT: "The environment",
        ChoiceGroup.HYPERPARAMETERS: "The hyperparameters",
    }
)


@dataclass(frozen=True, slots=True)
class ConfigDifference:
    """Uma escolha que muda entre os runs, escrita para cada um (``—`` se não a tiver).

    Attributes:
        label: O nome completo da escolha ("Hiperparâmetro · Taxa de aprendizagem").
        values: O valor em cada run, pela ordem dos runs.
        group: A parte do run a que pertence.
        name: O nome dentro do grupo ("Taxa de aprendizagem"), para mostrar por baixo do nome
            do grupo.
    """

    label: str
    values: tuple[str, ...]
    group: ChoiceGroup
    name: str


@dataclass(frozen=True, slots=True)
class _Choice:
    """Uma escolha de um run: o grupo, o nome dentro dele, o valor escrito e o prefixo do nome."""

    group: ChoiceGroup
    name: str
    value: str
    prefix: str | None = None

    @property
    def label(self) -> str:
        """O nome completo, com o prefixo ("Hiperparâmetro · Taxa de aprendizagem")."""
        return self.name if self.prefix is None else f"{self.prefix} · {self.name}"


@dataclass(frozen=True, slots=True)
class LossSeries:
    """A curva da loss de um run (no máximo ``LOSS_POINTS`` pontos)."""

    label: str
    steps: tuple[int, ...]
    losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Dois ou mais runs lado a lado.

    Attributes:
        runs: As entradas dos runs, pela ordem pedida.
        labels: Um nome curto para cada run ("A · Regressão logística em Iris").
        partition: A parte dos dados das métricas (o teste, se todos o tiverem).
        metrics: Cada métrica em cada run.
        differences: Só as escolhas que mudam entre os runs.
        losses: A curva da loss de cada run que treinou por passos.
        notices: A leitura da comparação.
    """

    runs: tuple[RunEntry, ...]
    labels: tuple[str, ...]
    partition: Partition
    metrics: tuple[MetricComparison, ...]
    differences: tuple[ConfigDifference, ...]
    losses: tuple[LossSeries, ...]
    notices: tuple[Notice, ...] = ()


def compare_runs(entries: Sequence[RunEntry], results: Sequence[Compared]) -> RunComparison:
    """Compara os runs (as entradas e os resultados vêm pela mesma ordem)."""
    labels = tuple(f"{chr(ord('A') + i)} · {entry.name}" for i, entry in enumerate(entries))
    has_test = all(result.main_partition is Partition.TEST for result in results)
    partition = Partition.TEST if has_test else Partition.TRAIN
    metrics = _metrics([result.reports.get(partition) for result in results])
    differences = _differences([_choices(result) for result in results])
    losses = tuple(
        _loss(label, result)
        for label, result in zip(labels, results, strict=True)
        if result.history
    )
    return RunComparison(
        runs=tuple(entries),
        labels=labels,
        partition=partition,
        metrics=metrics,
        differences=differences,
        losses=losses,
        notices=_reading(results, metrics, differences, labels, partition),
    )


def _metrics(reports: Sequence[EvaluationReport | None]) -> tuple[MetricComparison, ...]:
    """Todas as métricas que algum run tem, pela ordem em que aparecem."""
    first: dict[str, MetricResult] = {}
    for report in reports:
        for metric in report.metrics if report is not None else ():
            first.setdefault(metric.name, metric)
    rows = []
    for name, metric in first.items():
        values = tuple(_value(report, name) for report in reports)
        rows.append(
            MetricComparison(
                name=name,
                display_name=metric.display_name,
                as_percent=metric.as_percent,
                higher_is_better=metric.higher_is_better,
                values=values,
                best=_best(values, metric.higher_is_better),
            )
        )
    return tuple(rows)


def _value(report: EvaluationReport | None, name: str) -> float | None:
    found = None if report is None else report.find(name)
    return None if found is None else found.value


def _best(values: Sequence[float | None], higher_is_better: bool | None) -> int | None:
    """A posição do melhor valor, se houver um só melhor (num empate, nenhum)."""
    known = [(value, position) for position, value in enumerate(values) if value is not None]
    if higher_is_better is None or len(known) < 2:
        return None
    best = max(v for v, _ in known) if higher_is_better else min(v for v, _ in known)
    winners = [position for value, position in known if value == best]
    return winners[0] if len(winners) == 1 else None


def _choices(result: Compared) -> list[_Choice]:
    """Todas as escolhas de um run, escritas como texto, com o nome a mostrar."""
    if isinstance(result, ReinforcementResult):
        return _agent_choices(result)
    choices = [
        _Choice(ChoiceGroup.RUN, "Dataset", result.dataset_name),
        _Choice(ChoiceGroup.RUN, "Task", result.task_type.label),
        _Choice(ChoiceGroup.RUN, MODEL_CHOICE, result.model_display_name),
        _Choice(ChoiceGroup.RUN, SEED_CHOICE, str(result.seed)),
    ]
    titles = _titles(lambda: MODELS.get(result.model_name).info.hyperparameters)
    choices += [
        _Choice(ChoiceGroup.HYPERPARAMETERS, titles.get(name, name), _text(value), "Hyperparameter")
        for name, value in result.hyperparameters.items()
    ]
    return choices + _preparation_choices(result.config.preparation)


def _preparation_choices(preparation: PreparationConfig) -> list[_Choice]:
    """As escolhas da preparação, com o nome a mostrar de cada uma (a tarefa já está em cima)."""
    choices = [
        _Choice(ChoiceGroup.PREPARATION, "Target", _text(preparation.target)),
        _Choice(ChoiceGroup.PREPARATION, "Cleaning", _cleaning_text(preparation.cleaning)),
    ]
    parts = (
        ("Split", preparation.split),
        ("Transformations", preparation.transforms),
        ("Projection", preparation.projection),
    )
    for part, config in parts:
        for item in fields(config):
            title = _PREPARATION_FIELDS.get(item.name, item.name)
            # A parte fica no nome: dentro do grupo, "Split · strategy" ainda diz de onde vem.
            value = _text(getattr(config, item.name))
            choices.append(_Choice(ChoiceGroup.PREPARATION, f"{part} · {title}", value))
    return choices


def _cleaning_text(steps: Sequence[CleaningStep]) -> str:
    """A receita de limpeza por extenso: o nome de cada operação e os seus parâmetros."""
    if not steps:
        return "none"
    written = []
    for step in steps:
        try:
            title = CLEANING.get(step.operation).title
        except ComponentNotFoundError:  # uma operação que já não existe: fica o nome técnico
            title = step.operation
        params = ", ".join(_text(value) for value in step.params.values())
        written.append(f"{title} ({params})" if params else title)
    return "; ".join(written)


def _agent_choices(result: ReinforcementResult) -> list[_Choice]:
    """As escolhas de um run de reinforcement learning: o ambiente, o agente e a seed."""
    environment = result.config.environment.name
    try:
        environment_name = ENVIRONMENTS.get(environment).info.display_name
    except ComponentNotFoundError:
        environment_name = environment
    choices = [
        _Choice(ChoiceGroup.RUN, "Environment", environment_name),
        _Choice(ChoiceGroup.RUN, "Task", result.task_type.label),
        _Choice(ChoiceGroup.RUN, AGENT_CHOICE, result.agent_display_name),
        _Choice(ChoiceGroup.RUN, SEED_CHOICE, str(result.seed)),
    ]
    parameter_titles = _titles(lambda: ENVIRONMENTS.get(environment).info.parameters)
    choices += [
        _Choice(
            ChoiceGroup.ENVIRONMENT, parameter_titles.get(name, name), _text(value), "Environment"
        )
        for name, value in result.environment_parameters.items()
    ]
    titles = _titles(lambda: AGENTS.get(result.agent_name).info.hyperparameters)
    choices += [
        _Choice(ChoiceGroup.HYPERPARAMETERS, titles.get(name, name), _text(value), "Hyperparameter")
        for name, value in result.hyperparameters.items()
    ]
    return choices


def _titles(form: Callable[[], type[Hyperparameters]]) -> dict[str, str]:
    """O título de cada campo de um formulário, como a interface o mostra.

    Args:
        form: Uma função que devolve a classe do formulário (os hiperparâmetros de um modelo,
            por exemplo). Se o componente já não existir, ficam os nomes técnicos.
    """
    try:
        return form().titles()
    except ComponentNotFoundError:
        return {}


def _text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, Enum):  # antes do texto: um StrEnum também é um str
        return str(getattr(value, "label", value.value))
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format_significant(value)
    if isinstance(value, int | str):
        return str(value)
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return ", ".join(value) if value else "—"
    return json.dumps(to_json(value), ensure_ascii=False)


def _differences(choices: Sequence[Sequence[_Choice]]) -> tuple[ConfigDifference, ...]:
    """Só as escolhas que não são iguais em todos os runs, arrumadas pelo grupo.

    Dentro de cada grupo ficam pela ordem em que aparecem: com dois modelos diferentes, os
    hiperparâmetros de um vêm todos antes dos do outro (antes iam uns no meio da preparação).
    """
    runs = [{choice.label: choice for choice in run} for run in choices]
    first: dict[str, _Choice] = {}
    for run in runs:
        for label, choice in run.items():
            first.setdefault(label, choice)
    groups = list(ChoiceGroup)
    found = []
    # A ordenação é estável: dentro de um grupo fica a ordem em que as escolhas apareceram.
    for label, choice in sorted(first.items(), key=lambda item: groups.index(item[1].group)):
        values = tuple(run[label].value if label in run else "—" for run in runs)
        if len(set(values)) > 1:
            found.append(ConfigDifference(label, values, choice.group, choice.name))
    return tuple(found)


def _loss(label: str, result: Compared) -> LossSeries:
    points: list[tuple[int, float]] = [
        (record.step, record.loss) for record in result.history if record.loss is not None
    ]
    if len(points) > LOSS_POINTS:
        chosen = np.unique(np.linspace(0, len(points) - 1, LOSS_POINTS).astype(int))
        points = [points[index] for index in chosen]
    return LossSeries(label, tuple(step for step, _ in points), tuple(loss for _, loss in points))


def _reading(
    results: Sequence[Compared],
    metrics: Sequence[MetricComparison],
    differences: Sequence[ConfigDifference],
    labels: Sequence[str],
    partition: Partition,
) -> tuple[Notice, ...]:
    """O que a comparação diz, com as regras do início do módulo."""
    if len({result.task_type for result in results}) > 1:
        return (
            Notice.warning(
                "different_tasks",
                "The runs solve different tasks: their metrics don't compare directly.",
            ),
        )
    changed = [difference.label for difference in differences]
    if not changed:
        if all(len(set(metric.values)) == 1 for metric in metrics):
            return (
                Notice.info(
                    "reproduced",
                    "The runs have the same configuration and the same seed, and the metrics are "
                    "the same: the result reproduces.",
                ),
            )
        return (
            Notice.warning(
                "not_reproduced",
                "The runs have the same configuration and the same seed, but the metrics differ: "
                "something in the training isn't deterministic, or the data changed.",
            ),
        )
    notices = []
    if changed == [SEED_CHOICE]:
        notices.append(
            Notice.info(
                "seed_only",
                "Only the seed changes: the difference in the metrics is the effect of chance "
                "(another split and another initialisation). It gives an idea of how much a "
                "result varies without changing anything.",
            )
        )
    elif MODEL_CHOICE in changed or AGENT_CHOICE in changed:
        noun = "agents" if AGENT_CHOICE in changed else "models"
        notices.append(
            Notice.info(
                "different_models",
                f"The runs use different {noun}: the comparison measures each one with its own "
                "hyperparameters, not a single choice.",
            )
        )
    elif len(changed) > 1:
        notices.append(
            Notice.info(
                "several_changes",
                f"{len(changed)} choices change between the runs: there's no telling which one "
                "made the difference. To learn from the comparison, change one thing at a time.",
            )
        )
    primary = PRIMARY_METRIC.get(results[0].task_type)
    row = next((metric for metric in metrics if metric.name == primary), None)
    if row is not None and row.best is not None:
        value = row.values[row.best]
        assert value is not None  # o melhor tem sempre valor
        written = format_percent(value) if row.as_percent else format_number(value, 3)
        notices.append(
            Notice.info(
                "best",
                f"{row.display_name} on the {partition.label.lower()} data: the best is "
                f"{labels[row.best]} ({written}).",
            )
        )
    return tuple(notices)
