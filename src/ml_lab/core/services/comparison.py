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
from enum import Enum
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
    "strategy": "estratégia",
    "test_size": "parte de teste",
    "validation_size": "parte de validação",
    "time_column": "coluna do tempo",
    # As transformações.
    "numeric_imputation": "nulos numéricos",
    "categorical_imputation": "nulos categóricos",
    "scaling": "escala",
    "date_features": "features das datas",
    "polynomial_degree": "grau polinomial",
    "binned_columns": "colunas em intervalos",
    "n_bins": "número de intervalos",
    # A projeção 2D.
    "method": "método",
    "columns": "colunas",
}
"""O nome a mostrar de cada escolha da preparação (o resto do nome é a parte a que pertence)."""


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


@dataclass(frozen=True, slots=True)
class ConfigDifference:
    """Uma escolha que muda entre os runs, escrita para cada um (``—`` se não a tiver)."""

    label: str
    values: tuple[str, ...]


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


def _choices(result: Compared) -> dict[str, str]:
    """Todas as escolhas de um run, escritas como texto, pelo nome a mostrar."""
    if isinstance(result, ReinforcementResult):
        return _agent_choices(result)
    choices = {
        "Dataset": result.dataset_name,
        "Tarefa": result.task_type.label,
        "Modelo": result.model_display_name,
        "Seed": str(result.seed),
    }
    titles = _titles(lambda: MODELS.get(result.model_name).info.hyperparameters)
    for name, value in result.hyperparameters.items():
        choices[f"Hiperparâmetro · {titles.get(name, name)}"] = _text(value)
    return choices | _preparation_choices(result.config.preparation)


def _preparation_choices(preparation: PreparationConfig) -> dict[str, str]:
    """As escolhas da preparação, com o nome a mostrar de cada uma (a tarefa já está em cima)."""
    choices = {
        "Alvo": _text(preparation.target),
        "Limpeza": _cleaning_text(preparation.cleaning),
    }
    parts = (
        ("Divisão", preparation.split),
        ("Transformações", preparation.transforms),
        ("Projeção", preparation.projection),
    )
    for part, config in parts:
        for item in fields(config):
            title = _PREPARATION_FIELDS.get(item.name, item.name)
            choices[f"{part} · {title}"] = _text(getattr(config, item.name))
    return choices


def _cleaning_text(steps: Sequence[CleaningStep]) -> str:
    """A receita de limpeza por extenso: o nome de cada operação e os seus parâmetros."""
    if not steps:
        return "nenhuma"
    written = []
    for step in steps:
        try:
            title = CLEANING.get(step.operation).title
        except ComponentNotFoundError:  # uma operação que já não existe: fica o nome técnico
            title = step.operation
        params = ", ".join(_text(value) for value in step.params.values())
        written.append(f"{title} ({params})" if params else title)
    return "; ".join(written)


def _agent_choices(result: ReinforcementResult) -> dict[str, str]:
    """As escolhas de um run de reinforcement learning: o ambiente, o agente e a seed."""
    environment = result.config.environment.name
    try:
        environment_name = ENVIRONMENTS.get(environment).info.display_name
    except ComponentNotFoundError:
        environment_name = environment
    choices = {
        "Ambiente": environment_name,
        "Tarefa": result.task_type.label,
        "Agente": result.agent_display_name,
        "Seed": str(result.seed),
    }
    parameter_titles = _titles(lambda: ENVIRONMENTS.get(environment).info.parameters)
    for name, value in result.environment_parameters.items():
        choices[f"Ambiente · {parameter_titles.get(name, name)}"] = _text(value)
    titles = _titles(lambda: AGENTS.get(result.agent_name).info.hyperparameters)
    for name, value in result.hyperparameters.items():
        choices[f"Hiperparâmetro · {titles.get(name, name)}"] = _text(value)
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
        return "sim" if value else "não"
    if isinstance(value, float):
        return format_significant(value)
    if isinstance(value, int | str):
        return str(value)
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return ", ".join(value) if value else "—"
    return json.dumps(to_json(value), ensure_ascii=False)


def _differences(choices: Sequence[Mapping[str, str]]) -> tuple[ConfigDifference, ...]:
    """Só as escolhas que não são iguais em todos os runs."""
    labels = list(dict.fromkeys(label for run in choices for label in run))
    found = []
    for label in labels:
        values = tuple(run.get(label, "—") for run in choices)
        if len(set(values)) > 1:
            found.append(ConfigDifference(label, values))
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
                "Os runs resolvem tarefas diferentes: as métricas não se comparam diretamente.",
            ),
        )
    changed = [difference.label for difference in differences]
    if not changed:
        if all(len(set(metric.values)) == 1 for metric in metrics):
            return (
                Notice.info(
                    "reproduced",
                    "Os runs têm a mesma configuração e a mesma seed, e as métricas são iguais: "
                    "o resultado reproduz-se.",
                ),
            )
        return (
            Notice.warning(
                "not_reproduced",
                "Os runs têm a mesma configuração e a mesma seed, mas as métricas diferem: "
                "alguma coisa no treino não é determinística, ou os dados mudaram.",
            ),
        )
    notices = []
    if changed == ["Seed"]:
        notices.append(
            Notice.info(
                "seed_only",
                "Só a seed muda: a diferença nas métricas é o efeito do acaso (outra divisão e "
                "outra inicialização). Dá uma ideia de quanto um resultado varia sem mudar nada.",
            )
        )
    elif "Modelo" in changed or "Agente" in changed:
        noun = "agentes" if "Agente" in changed else "modelos"
        notices.append(
            Notice.info(
                "different_models",
                f"Os runs usam {noun} diferentes: a comparação mede cada um com os seus "
                "hiperparâmetros, e não uma escolha só.",
            )
        )
    elif len(changed) > 1:
        notices.append(
            Notice.info(
                "several_changes",
                f"Mudam {len(changed)} escolhas entre os runs: não se sabe qual delas fez a "
                "diferença. Para aprender com a comparação, muda uma coisa de cada vez.",
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
                f"{row.display_name} no {partition.label.lower()}: o melhor é "
                f"{labels[row.best]} ({written}).",
            )
        )
    return tuple(notices)
