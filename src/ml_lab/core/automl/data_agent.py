"""O agente de dados: prepara a tabela para os modelos, a partir do perfil, e diz porquê.

Faz o que se faz à mão no ecrã Preparar, pela mesma ordem:

1. **limpeza**: aceita as sugestões do perfil (linhas duplicadas, números e datas
   guardados como texto, identificadores, texto livre, colunas vazias ou quase);
2. **tarefa**: a que a deteção sugere para o alvo, ou a que se pediu;
3. **divisão**: 20% para o teste, estratificada quando há classes. O teste fica guardado
   até ao fim: só o vencedor o vê;
4. **transformações**: os nulos preenchem-se (mediana nas numéricas, o valor mais frequente
   nas categóricas), as categóricas passam a one-hot e as numéricas são padronizadas;
5. **objetivo**: a métrica principal da tarefa, ou a exatidão equilibrada quando as classes
   estão desequilibradas e ninguém pediu outra.

São regras, não um modelo de linguagem: com os mesmos dados, decide sempre o mesmo, e cada
decisão traz a razão. O agente nunca olha para os valores das linhas, só para o perfil (o
pandas fica na camada de dados). Se já houver uma preparação feita à mão, adota-a.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ml_lab.core.data.cleaning.base import CLEANING, CleaningStep
from ml_lab.core.data.cleaning.suggestions import suggest_cleaning
from ml_lab.core.data.preparation import PreparationConfig, task_options
from ml_lab.core.data.profiling.column_profile import ColumnProfile
from ml_lab.core.data.profiling.dataset_profile import DatasetProfile
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.task.task_detection import TaskSuggestion
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig
from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.metrics.base import METRICS
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.core.foundation.notices import NoticeLevel
from ml_lab.core.foundation.types import TaskType
from ml_lab.formatting import format_list, format_percent

TEST_SIZE: Final = 0.2
"""A parte das linhas que fica para o teste."""

_CONVERSIONS: Final = {"numeric": "a number", "datetime": "a date", "boolean": "a boolean"}


@dataclass(frozen=True, slots=True)
class DataDecision:
    """Uma decisão do agente de dados.

    Attributes:
        title: O que decidiu ("Drop PassengerId").
        reason: Porquê (a mostrar na interface).
    """

    title: str
    reason: str


@dataclass(frozen=True, slots=True)
class DataPlan:
    """O que o agente de dados decidiu para um pedido.

    Attributes:
        target: A coluna-alvo.
        task_type: A tarefa.
        preparation: A preparação, pronta para a camada de dados.
        objective: A métrica que sugere otimizar.
        decisions: Cada decisão, pela ordem, com a razão.
        adopted: Se adotou uma preparação feita à mão (em vez de decidir).
    """

    target: str
    task_type: TaskType
    preparation: PreparationConfig
    objective: str
    decisions: tuple[DataDecision, ...]
    adopted: bool = False


class DataAgent:
    """Decide a preparação dos dados a partir do perfil."""

    def plan(
        self,
        profile: DatasetProfile,
        task: TaskSuggestion,
        *,
        target: str,
        task_type: TaskType | None = None,
        preparation: PreparationConfig | None = None,
        objective: str | None = None,
    ) -> DataPlan:
        """A preparação para prever ``target``, com a razão de cada decisão.

        Args:
            profile: O perfil do dataset (com o alvo, para se ver o equilíbrio das classes).
            task: A sugestão de tarefa para o alvo.
            target: A coluna-alvo.
            task_type: A tarefa pedida; ``None`` aceita a sugestão.
            preparation: Uma preparação feita à mão, que o agente adota tal como está.
            objective: A métrica pedida; ``None`` deixa o agente sugerir.

        Raises:
            DataError: Se o alvo não servir para prever (um só valor, por exemplo).
            InvalidParameterError: Se a tarefa pedida não servir para o alvo, ou a
                preparação feita à mão for de outro alvo.
        """
        errors = [notice.message for notice in task.notices if notice.level is NoticeLevel.ERROR]
        if errors or task.task_type is None:
            raise DataError(" ".join(errors) or task.reason)
        if preparation is not None:
            return self._adopt(profile, task, target, preparation, objective)
        chosen = task_type or task.task_type
        if chosen not in task.allowed:
            allowed = ", ".join(option.label.lower() for option in task.allowed)
            raise InvalidParameterError(
                f"{chosen.label} doesn't suit the target {target}. Choose one of these: {allowed}."
            )
        decisions: list[DataDecision] = []
        cleaning = []
        for suggestion in suggest_cleaning(profile, target):
            cleaning.append(suggestion.step)
            decisions.append(DataDecision(_cleaning_title(suggestion.step), suggestion.reason))
        decisions.append(_task_decision(task, chosen))
        split = _split(chosen, target)
        decisions.append(_split_decision(split))
        features = _features(profile, target, cleaning)
        decisions.extend(_transform_decisions(features))
        suggested, reason = _objective(profile, chosen, objective)
        decisions.append(DataDecision(f"Optimise: {METRICS.get(suggested).display_name}", reason))
        config = PreparationConfig(
            target=target,
            task_type=chosen,
            cleaning=tuple(cleaning),
            split=split,
            transforms=TransformConfig(),
        )
        return DataPlan(target, chosen, config, suggested, tuple(decisions))

    def _adopt(
        self,
        profile: DatasetProfile,
        task: TaskSuggestion,
        target: str,
        preparation: PreparationConfig,
        objective: str | None,
    ) -> DataPlan:
        if preparation.target != target:
            raise InvalidParameterError(
                f"The hand-made preparation is for the target {preparation.target}, and you "
                f"asked for {target}: prepare again in Prepare, or let the agent decide."
            )
        chosen = preparation.task_type or task.task_type
        assert chosen is not None  # confirmado em plan
        suggested, reason = _objective(profile, chosen, objective)
        steps = len(preparation.cleaning)
        decisions = (
            DataDecision(
                "Use the preparation from Prepare",
                f"You've already prepared this data by hand ({steps} cleaning "
                f"step{'s' if steps != 1 else ''}, the split and the transformations you chose): "
                "the data agent respects your choices and decides nothing on top of them.",
            ),
            _task_decision(task, chosen),
            DataDecision(f"Optimise: {METRICS.get(suggested).display_name}", reason),
        )
        config = PreparationConfig(
            target=target,
            task_type=chosen,
            cleaning=preparation.cleaning,
            split=preparation.split,
            transforms=preparation.transforms,
            projection=preparation.projection,
        )
        return DataPlan(target, chosen, config, suggested, decisions, adopted=True)


def _cleaning_title(step: CleaningStep) -> str:
    params = step.params
    if step.operation == "drop_columns":
        return f"Drop {', '.join(str(column) for column in params.get('columns', ()))}"
    if step.operation == "convert_type":
        to = _CONVERSIONS.get(str(params.get("to")), str(params.get("to")))
        return f"Convert {params.get('column')} to {to}"
    return CLEANING.get(step.operation).title


def _task_decision(task: TaskSuggestion, chosen: TaskType) -> DataDecision:
    title = f"Task: {chosen.label.lower()}"
    if chosen is task.task_type:
        return DataDecision(title, task.reason)
    assert task.task_type is not None
    return DataDecision(
        title,
        f"It was the task asked for. The detection suggested {task.task_type.label.lower()}: "
        f"{task.reason}",
    )


def _split(task_type: TaskType, target: str) -> SplitConfig:
    strategy = task_options(task_type, target).split_strategies[0]
    return SplitConfig(strategy=strategy, test_size=TEST_SIZE)


def _split_decision(split: SplitConfig) -> DataDecision:
    train = format_percent(1 - split.test_size, 0)
    test = format_percent(split.test_size, 0)
    title = f"{split.strategy.label} split: {train} for training, {test} for test"
    if split.strategy is SplitStrategy.STRATIFIED:
        how = (
            "With classes, the test keeps the same share of each one: at random, a rare class "
            "could almost go missing from the test."
        )
    else:
        how = "Without classes in the target, the rows go to the training or the test at random."
    keep = (
        " The test stays put away until the end: the agents choose the model with the training "
        "data only, and only the winner is evaluated on the test."
    )
    return DataDecision(title, how + keep)


def _features(
    profile: DatasetProfile, target: str, cleaning: Sequence[CleaningStep]
) -> list[ColumnProfile]:
    """As colunas que chegam ao modelo: as que ficam depois da limpeza, sem o alvo."""
    dropped = {
        str(column)
        for step in cleaning
        if step.operation == "drop_columns"
        for column in step.params.get("columns", ())
    }
    return [
        column
        for column in profile.columns
        if column.name != target and column.name not in dropped and column.kind.is_usable_by_models
    ]


def _transform_decisions(features: Sequence[ColumnProfile]) -> list[DataDecision]:
    decisions = []
    numeric = [c.name for c in features if c.kind is ColumnKind.NUMERIC]
    categorical = [c.name for c in features if c.kind is ColumnKind.CATEGORICAL]
    dates = [c.name for c in features if c.kind is ColumnKind.DATETIME]
    numeric_gaps = [c.name for c in features if c.kind is ColumnKind.NUMERIC and c.n_missing]
    other_gaps = [
        c.name
        for c in features
        if c.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN) and c.n_missing
    ]
    if dates:
        decisions.append(
            DataDecision(
                f"Split the dates of {format_list(dates)}",
                "A model can't read a date: it becomes the year, month, day and day of the week, "
                "which are numbers it can learn from (December sales, weekends).",
            )
        )
    if numeric_gaps:
        decisions.append(
            DataDecision(
                f"Fill in the nulls of {format_list(numeric_gaps)} with the median",
                "Models don't accept missing values. The median is learned from the training "
                "data only and isn't pulled by extreme values, as the mean is.",
            )
        )
    if other_gaps:
        decisions.append(
            DataDecision(
                f"Fill in the nulls of {format_list(other_gaps)} with the most frequent value",
                "A column of categories has no mean: the most frequent value in the training "
                "data is the safest bet.",
            )
        )
    if categorical:
        decisions.append(
            DataDecision(
                f"One-hot encode {format_list(categorical)}",
                "Models only take numbers: each category becomes a column of 0s and 1s, without "
                "making up an order between them (giving 1, 2 and 3 to three ports would say "
                "that 3 is three times 1).",
            )
        )
    if numeric:
        decisions.append(
            DataDecision(
                "Standardise the numeric columns (mean 0, standard deviation 1)",
                "Trees don't need it, but KNN, SVM, logistic regression and networks do: without "
                "scaling, the feature with the biggest numbers rules the distances and the "
                "gradients. As the same preparation serves every model, everything is scaled.",
            )
        )
    return decisions


def _objective(
    profile: DatasetProfile, task_type: TaskType, requested: str | None
) -> tuple[str, str]:
    """A métrica a otimizar e porquê.

    Pedir a métrica que o agente sugeria não muda a razão (é o caso de quem aceita a
    sugestão num formulário); pedir outra fica dito, com o que o agente sugeria e porquê.
    """
    suggested, reason = _suggested_objective(profile, task_type)
    if requested is None or requested == suggested:
        return suggested, reason
    label = METRICS.get(suggested).display_name
    return requested, f"It was the metric asked for (the data agent suggested {label}). {reason}"


def _suggested_objective(profile: DatasetProfile, task_type: TaskType) -> tuple[str, str]:
    """A métrica que o agente de dados sugere, e porquê."""
    primary = PRIMARY_METRIC[task_type]
    balance = profile.class_balance
    if task_type.is_classification and balance is not None and balance.is_imbalanced:
        rarest = balance.classes[-1]
        return "balanced_accuracy", (
            f"The classes are imbalanced: the class “{rarest.value}” only has "
            f"{format_percent(rarest.ratio)} of the rows. A model that never predicts it would "
            "have a high accuracy and be no use at all; the balanced accuracy gives each class "
            "the same weight."
        )
    if task_type.is_classification and balance is not None:
        rarest = balance.classes[-1]
        return primary, (
            f"The classes are reasonably balanced (the rarest, “{rarest.value}”, has "
            f"{format_percent(rarest.ratio)} of the rows): the accuracy, the main metric, will do."
        )
    return primary, (
        f"It's the main metric of {task_type.label.lower()}: it says how much of the target's "
        "variation the model explains (1 is perfect; 0 is as good as always predicting the mean)."
        if primary == "r2"
        else f"It's the main metric of {task_type.label.lower()}."
    )
