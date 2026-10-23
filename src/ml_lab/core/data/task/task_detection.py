"""Sugere o TaskType a partir da coluna-alvo (a deteção é só uma sugestão).

As regras (secção "Regras de deteção automática do tipo de tarefa" do plano):

1. sem coluna-alvo, só aparecem clustering, redução de dimensionalidade e deteção
   de anomalias;
2. alvo categórico (texto, booleano ou inteiro com poucos valores distintos):
   classificação, binária com duas classes, multiclasse com mais;
3. alvo numérico contínuo: regressão;
4. o utilizador pode sempre escolher outro tipo entre os permitidos.
"""

from dataclasses import dataclass
from typing import Final

import pandas as pd

from ml_lab.core.data.profiling.type_inference import infer_column, looks_like_classes
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.foundation.errors import ColumnNotFoundError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.formatting import format_count, format_quantity

UNSUPERVISED_TASKS: Final = tuple(task for task in TaskType if not task.requires_target)
"""Tarefas possíveis sem coluna-alvo."""

MAX_INTEGER_CLASSES: Final = 50
"""Um alvo inteiro com até este número de valores ainda pode ser tratado como classes."""


@dataclass(frozen=True, slots=True)
class TaskSuggestion:
    """O tipo de tarefa sugerido para um alvo e as alternativas permitidas.

    Attributes:
        target: A coluna-alvo, ou ``None`` para aprendizagem não supervisionada.
        task_type: O tipo sugerido, ou ``None`` se o alvo não servir para nenhum.
        allowed: Os tipos que o utilizador pode escolher, com o sugerido em primeiro.
        reason: Porque se sugere este tipo, em português.
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
                "Sem coluna-alvo só há aprendizagem não supervisionada: agrupar os dados "
                "(clustering), reduzir as dimensões ou detetar anomalias."
            ),
        )
    if target not in frame.columns:
        raise ColumnNotFoundError(target, [str(name) for name in frame.columns])

    series = frame[target]
    present = series.dropna()
    notices: list[Notice] = []
    n_missing = len(series) - len(present)
    if n_missing:
        rows = format_quantity(n_missing, "linha", "linhas")
        notices.append(
            Notice.warning(
                "missing_target",
                f"O alvo {target} está em falta em {rows}, que a preparação remove: sem alvo, "
                "uma linha não serve para aprender.",
                target,
            )
        )
    if present.empty:
        return _unusable(target, f"O alvo {target} não tem nenhum valor.", notices)
    n_unique = int(present.nunique())
    if n_unique == 1:
        return _unusable(
            target, f"O alvo {target} tem sempre o mesmo valor: não há nada para prever.", notices
        )

    inference = infer_column(series)
    if inference.kind is ColumnKind.DATETIME:
        return _unusable(
            target,
            f"{target} é uma coluna de datas, que não serve de alvo. Escolhe outra coluna (ou "
            "extrai uma parte da data, como o ano, para uma coluna numérica).",
            notices,
        )
    if looks_like_classes(series, inference):
        return _classification(target, present, inference.kind, n_unique, notices)
    if inference.kind is ColumnKind.NUMERIC:
        return _regression(target, present, inference.stored_as_text, n_unique, notices)
    notices.append(
        Notice.warning(
            "many_classes",
            f"O alvo {target} tem {format_count(n_unique)} valores diferentes: parece texto "
            "livre ou um identificador. Um classificador precisa de muitos exemplos de cada "
            "classe para aprender.",
            target,
        )
    )
    return TaskSuggestion(
        target=target,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        allowed=(TaskType.MULTICLASS_CLASSIFICATION,),
        reason=f"O alvo {target} é texto, por isso cada valor diferente é uma classe.",
        n_classes=n_unique,
        notices=tuple(notices),
    )


def _classification(
    target: str, present: pd.Series, kind: ColumnKind, n_classes: int, notices: list[Notice]
) -> TaskSuggestion:
    task = TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    label = task.label.lower()
    if kind is ColumnKind.BOOLEAN:
        values = " e ".join(sorted(present.astype(str).unique().tolist()))
        reason = f"O alvo {target} só tem dois valores ({values}): {label}."
    elif kind is ColumnKind.CATEGORICAL:
        reason = f"O alvo {target} é categórico, com {n_classes} classes: {label}."
    else:
        reason = (
            f"O alvo {target} é um número inteiro com poucos valores distintos ({n_classes}), "
            f"por isso cada valor é uma classe: {label}. Se os números forem quantidades "
            "(e não categorias), escolhe regressão."
        )
    allowed = (task, TaskType.REGRESSION) if kind is ColumnKind.NUMERIC else (task,)
    counts = present.astype(str).value_counts()
    tiny = [str(value) for value, count in counts.items() if count < 2]
    if tiny:
        notices.append(
            Notice.warning(
                "tiny_class",
                f"As classes {', '.join(tiny)} de {target} só têm uma linha: não chegam ao treino "
                "e ao teste ao mesmo tempo.",
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
                f"O alvo {target} são números guardados como texto: converte-o para número na "
                "limpeza (operação convert_type) antes de preparar os dados.",
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
        f"O alvo {target} é numérico, com {format_count(n_unique)} valores distintos: "
        "regressão (prever um número)."
    )
    return TaskSuggestion(target, TaskType.REGRESSION, allowed, reason, None, tuple(notices))


def _unusable(target: str, message: str, notices: list[Notice]) -> TaskSuggestion:
    notices.append(Notice.error("unusable_target", message, target))
    return TaskSuggestion(target, None, (), message, notices=tuple(notices))
