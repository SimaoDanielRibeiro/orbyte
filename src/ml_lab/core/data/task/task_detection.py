"""Sugere o TaskType a partir da coluna-alvo (a deteção é só uma sugestão).

As regras (secção "Regras de deteção automática do tipo de tarefa" do plano):

1. sem coluna-alvo, só aparecem clustering, redução de dimensionalidade e deteção
   de anomalias;
2. alvo categórico (texto, booleano ou inteiro com poucos valores distintos):
   classificação, binária com duas classes, multiclasse com mais;
3. alvo numérico contínuo: regressão;
4. o utilizador pode sempre escolher outro tipo entre os permitidos. Com um alvo
   categórico, também as tarefas sem alvo: a coluna fica como referência, só para
   comparar no fim (por exemplo, os grupos do k-means com as espécies do Iris).
"""

from dataclasses import dataclass
from typing import Final

import pandas as pd

from ml_lab.core.data.profiling.type_inference import infer_column, looks_like_classes
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.foundation.errors import ColumnNotFoundError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Paradigm, TaskType
from ml_lab.formatting import format_count, format_quantity

UNSUPERVISED_TASKS: Final = tuple(
    task for task in TaskType if task.paradigm is Paradigm.UNSUPERVISED
)
"""Tarefas possíveis sem coluna-alvo (o controlo não aprende com dados: fica de fora)."""

MAX_INTEGER_CLASSES: Final = 50
"""Um alvo inteiro com até este número de valores ainda pode ser tratado como classes."""


@dataclass(frozen=True, slots=True)
class TaskSuggestion:
    """O tipo de tarefa sugerido para um alvo e as alternativas permitidas.

    Attributes:
        target: A coluna-alvo, ou ``None`` para aprendizagem não supervisionada.
        task_type: O tipo sugerido, ou ``None`` se o alvo não servir para nenhum.
        allowed: Os tipos que o utilizador pode escolher, com o sugerido em primeiro.
        reason: Porque se sugere este tipo, em inglês.
        n_classes: Número de classes, nas tarefas de classificação.
        notices: Avisos sobre o alvo; um erro impede usá-lo.
    """

    target: str | None
    task_type: TaskType | None
    allowed: tuple[TaskType, ...]
    reason: str
    n_classes: int | None = None
    notices: tuple[Notice, ...] = ()


def suggest_task(frame: pd.DataFrame, target: str | None) -> TaskSuggestion:
    """Sugere o tipo de tarefa para ``target``.

    Args:
        frame: A tabela (normalmente já limpa).
        target: A coluna-alvo, ou ``None`` se não houver.

    Raises:
        ColumnNotFoundError: Se ``target`` não existir.
    """
    if target is None:
        return TaskSuggestion(
            target=None,
            task_type=TaskType.CLUSTERING,
            allowed=UNSUPERVISED_TASKS,
            reason=(
                "Without a target column there's only unsupervised learning: grouping the data "
                "(clustering), reducing its dimensions or detecting anomalies."
            ),
        )
    if target not in frame.columns:
        raise ColumnNotFoundError(target, [str(name) for name in frame.columns])

    series = frame[target]
    present = series.dropna()
    notices: list[Notice] = []
    n_missing = len(series) - len(present)
    if n_missing:
        rows = format_quantity(n_missing, "row", "rows")
        notices.append(
            Notice.warning(
                "missing_target",
                f"The target {target} is missing in {rows}, which the preparation drops: without "
                "a target, a row is no use for learning.",
                target,
            )
        )
    if present.empty:
        return _unusable(target, f"The target {target} has no values at all.", notices)
    n_unique = int(present.nunique())
    if n_unique == 1:
        return _unusable(
            target,
            f"The target {target} always has the same value: there's nothing to predict.",
            notices,
        )

    inference = infer_column(series)
    if inference.kind is ColumnKind.DATETIME:
        return _unusable(
            target,
            f"{target} is a date column, which can't be a target. Choose another column (or "
            "extract part of the date, such as the year, into a numeric column).",
            notices,
        )
    if looks_like_classes(series, inference):
        return _classification(target, present, inference.kind, n_unique, notices)
    if inference.kind is ColumnKind.NUMERIC:
        return _regression(target, present, inference.stored_as_text, n_unique, notices)
    notices.append(
        Notice.warning(
            "many_classes",
            f"The target {target} has {format_count(n_unique)} different values: it looks like "
            "free text or an identifier. A classifier needs many examples of each class to "
            "learn.",
            target,
        )
    )
    return TaskSuggestion(
        target=target,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        allowed=(TaskType.MULTICLASS_CLASSIFICATION,),
        reason=f"The target {target} is text, so each different value is a class.",
        n_classes=n_unique,
        notices=tuple(notices),
    )


def _classification(
    target: str, present: pd.Series, kind: ColumnKind, n_classes: int, notices: list[Notice]
) -> TaskSuggestion:
    task = TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    label = task.label.lower()
    if kind is ColumnKind.BOOLEAN:
        values = " and ".join(sorted(present.astype(str).unique().tolist()))
        reason = f"The target {target} only has two values ({values}): {label}."
    elif kind is ColumnKind.CATEGORICAL:
        reason = f"The target {target} is categorical, with {n_classes} classes: {label}."
    else:
        reason = (
            f"The target {target} is a whole number with few distinct values ({n_classes}), so "
            f"each value is a class: {label}. If the numbers are quantities (and not "
            "categories), choose regression."
        )
    allowed = (task, TaskType.REGRESSION) if kind is ColumnKind.NUMERIC else (task,)
    # Um alvo com classes também serve de referência às tarefas sem alvo: não entra no
    # treino, mas diz se os grupos (ou as anomalias) encontrados correspondem às classes.
    allowed = (*allowed, *UNSUPERVISED_TASKS)
    counts = present.astype(str).value_counts()
    tiny = [str(value) for value, count in counts.items() if count < 2]
    if tiny:
        notices.append(
            Notice.warning(
                "tiny_class",
                f"The classes {', '.join(tiny)} of {target} only have one row: they can't reach "
                "both the training and the test data.",
                target,
            )
        )
    return TaskSuggestion(target, task, allowed, reason, n_classes, tuple(notices))


def _regression(
    target: str, present: pd.Series, stored_as_text: bool, n_unique: int, notices: list[Notice]
) -> TaskSuggestion:
    if stored_as_text:
        notices.append(
            Notice.error(
                "target_as_text",
                f"The target {target} is numbers stored as text: convert it to a number in the "
                "cleaning (the convert_type operation) before preparing the data.",
                target,
            )
        )
        integral = False
    else:
        integral = bool((present == present.round()).all())
    allowed: tuple[TaskType, ...] = (TaskType.REGRESSION,)
    if integral and n_unique <= MAX_INTEGER_CLASSES:
        allowed = (TaskType.REGRESSION, TaskType.MULTICLASS_CLASSIFICATION)
    reason = (
        f"The target {target} is numeric, with {format_count(n_unique)} distinct values: "
        "regression (predicting a number)."
    )
    return TaskSuggestion(target, TaskType.REGRESSION, allowed, reason, None, tuple(notices))


def _unusable(target: str, message: str, notices: list[Notice]) -> TaskSuggestion:
    notices.append(Notice.error("unusable_target", message, target))
    return TaskSuggestion(target, None, (), message, notices=tuple(notices))
