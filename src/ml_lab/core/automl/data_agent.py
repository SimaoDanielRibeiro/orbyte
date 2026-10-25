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

_CONVERSIONS: Final = {"numeric": "número", "datetime": "data", "boolean": "booleano"}


@dataclass(frozen=True, slots=True)
class DataDecision:
    """Uma decisão do agente de dados.

    Attributes:
        title: O que decidiu ("Remover PassengerId").
        reason: Porquê, em português.
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
                f"{chosen.label} não serve para o alvo {target}. Escolhe uma destas: {allowed}."
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
        decisions.append(DataDecision(f"Otimizar: {METRICS.get(suggested).display_name}", reason))
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
                f"A preparação feita à mão é para o alvo {preparation.target}, e pediste "
                f"{target}: prepara outra vez no ecrã Preparar, ou deixa o agente decidir."
            )
        chosen = preparation.task_type or task.task_type
        assert chosen is not None  # confirmado em plan
        suggested, reason = _objective(profile, chosen, objective)
        steps = len(preparation.cleaning)
        decisions = (
            DataDecision(
                "Usar a preparação do ecrã Preparar",
                f"Já preparaste estes dados à mão ({steps} passo{'s' if steps != 1 else ''} de "
                "limpeza, a divisão e as transformações que escolheste): o agente de dados "
                "respeita as tuas escolhas e não decide nada por cima.",
            ),
            _task_decision(task, chosen),
            DataDecision(f"Otimizar: {METRICS.get(suggested).display_name}", reason),
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
        return f"Remover {', '.join(str(column) for column in params.get('columns', ()))}"
    if step.operation == "convert_type":
        to = _CONVERSIONS.get(str(params.get("to")), str(params.get("to")))
        return f"Converter {params.get('column')} em {to}"
    return CLEANING.get(step.operation).title


def _task_decision(task: TaskSuggestion, chosen: TaskType) -> DataDecision:
    title = f"Tarefa: {chosen.label.lower()}"
    if chosen is task.task_type:
        return DataDecision(title, task.reason)
    assert task.task_type is not None
    return DataDecision(
        title,
        f"Foi a tarefa pedida. A deteção sugeria {task.task_type.label.lower()}: {task.reason}",
    )


def _split(task_type: TaskType, target: str) -> SplitConfig:
    strategy = task_options(task_type, target).split_strategies[0]
    return SplitConfig(strategy=strategy, test_size=TEST_SIZE)


def _split_decision(split: SplitConfig) -> DataDecision:
    train = format_percent(1 - split.test_size, 0)
    test = format_percent(split.test_size, 0)
    title = f"Divisão {split.strategy.label.lower()}: {train} para treino, {test} para teste"
    if split.strategy is SplitStrategy.STRATIFIED:
        how = (
            "Com classes, o teste fica com a mesma proporção de cada uma: ao acaso, uma "
            "classe rara podia quase faltar no teste."
        )
    else:
        how = "Sem classes no alvo, as linhas vão ao acaso para o treino ou para o teste."
    keep = (
        " O teste fica guardado até ao fim: os agentes escolhem o modelo só com o treino, e "
        "só o vencedor é avaliado no teste."
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
                f"Partir as datas de {format_list(dates)}",
                "Um modelo não lê uma data: passa a ano, mês, dia e dia da semana, que são "
                "números com que pode aprender (as vendas de dezembro, os fins de semana).",
            )
        )
    if numeric_gaps:
        decisions.append(
            DataDecision(
                f"Preencher os nulos de {format_list(numeric_gaps)} com a mediana",
                "Os modelos não aceitam valores em falta. A mediana aprende-se só com o treino e "
                "não se deixa puxar pelos valores extremos, como a média.",
            )
        )
    if other_gaps:
        decisions.append(
            DataDecision(
                f"Preencher os nulos de {format_list(other_gaps)} com o valor mais frequente",
                "Numa coluna de categorias não há média: o valor mais frequente no treino é a "
                "aposta mais segura.",
            )
        )
    if categorical:
        decisions.append(
            DataDecision(
                f"One-hot em {format_list(categorical)}",
                "Os modelos só recebem números: cada categoria passa a uma coluna de 0 e 1, sem "
                "inventar uma ordem entre elas (dar 1, 2 e 3 a três portos diria que o 3 é o "
                "triplo do 1).",
            )
        )
    if numeric:
        decisions.append(
            DataDecision(
                "Padronizar as numéricas (média 0, desvio padrão 1)",
                "As árvores não precisam, mas o KNN, o SVM, a regressão logística e as redes sim: "
                "sem escala, a feature com os números maiores manda nas distâncias e nos "
                "gradientes. Como a mesma preparação serve todos os modelos, escala-se.",
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
    return requested, f"Foi a métrica pedida (o agente de dados sugeria {label}). {reason}"


def _suggested_objective(profile: DatasetProfile, task_type: TaskType) -> tuple[str, str]:
    """A métrica que o agente de dados sugere, e porquê."""
    primary = PRIMARY_METRIC[task_type]
    balance = profile.class_balance
    if task_type.is_classification and balance is not None and balance.is_imbalanced:
        rarest = balance.classes[-1]
        return "balanced_accuracy", (
            f"As classes estão desequilibradas: a classe «{rarest.value}» só tem "
            f"{format_percent(rarest.ratio)} das linhas. Um modelo que nunca a prevê teria uma "
            "exatidão alta e não serviria para nada; a exatidão equilibrada dá o mesmo peso a "
            "cada classe."
        )
    if task_type.is_classification and balance is not None:
        rarest = balance.classes[-1]
        return primary, (
            f"As classes estão razoavelmente equilibradas (a mais rara, «{rarest.value}», tem "
            f"{format_percent(rarest.ratio)} das linhas): a exatidão, a métrica principal, serve."
        )
    return primary, (
        f"É a métrica principal de {task_type.label.lower()}: diz que parte da variação do "
        "alvo o modelo explica (1 é perfeito; 0 é tão bom como prever sempre a média)."
        if primary == "r2"
        else f"É a métrica principal de {task_type.label.lower()}."
    )
