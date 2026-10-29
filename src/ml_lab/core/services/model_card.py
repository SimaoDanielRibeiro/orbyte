"""A ficha de um modelo: o ``README.md`` de um pacote, escrito para pessoas (ADR 0039).

Uma ficha de modelo diz o que o modelo prevê, com que dados aprendeu, como se portou, onde não
se deve confiar nele e como se usa. É o que quem recebe o ficheiro lê primeiro, antes de
abrir os JSON. Os números vêm escritos à portuguesa, como em todo o laboratório.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.lineage import Stage
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.serving import FittedPreparation
from ml_lab.core.foundation.notices import NoticeLevel
from ml_lab.core.foundation.types import Partition
from ml_lab.core.models.base import ModelInfo
from ml_lab.core.services.experiments import ExperimentResult
from ml_lab.formatting import format_list, format_value

_PREPARATION_STAGES = (Stage.CLEANING, Stage.TASK, Stage.SPLIT, Stage.TRANSFORM)
"""As etapas da linhagem que a ficha conta (a entrada e a saída dizem-se noutros sítios)."""


@dataclass(frozen=True, slots=True)
class CardFile:
    """Um ficheiro do pacote, como a ficha o descreve.

    Attributes:
        path: O nome no pacote.
        description: O que tem, numa frase.
    """

    path: str
    description: str


def model_card(
    *,
    name: str,
    result: ExperimentResult,
    dataset: Dataset,
    preparation: FittedPreparation,
    info: ModelInfo,
    files: Sequence[CardFile],
    onnx_reason: str | None,
) -> str:
    """O ``README.md`` de um pacote.

    Args:
        name: O nome do pacote.
        result: O run.
        dataset: Os dados preparados do run.
        preparation: A preparação aprendida.
        info: A ficha técnica do modelo.
        files: Os ficheiros que vão no pacote.
        onnx_reason: Porque é que o pacote vai sem o ``model.onnx`` (``None`` se vai com ele).
    """
    sections = [
        f"# {name}",
        _summary(result, preparation, info),
        "## Como foi feito",
        _how(result, dataset, info),
        "## Como se portou",
        _performance(result),
        "## Os limites",
        _limits(result, preparation, info),
        "## O que vai neste pacote",
        _files(files),
        "## Como se usa",
        _usage(files, onnx_reason),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _summary(result: ExperimentResult, preparation: FittedPreparation, info: ModelInfo) -> str:
    columns = format_list([column.name for column in preparation.columns])
    task = result.task_type.label.lower()
    if preparation.target is not None and result.task_type.requires_target:
        what = f"prevê **{preparation.target}** ({task}) a partir de {columns}"
    else:
        what = f"faz {task} com {columns}"
    return f"{info.display_name}, treinado no ML Lab: {what}."


def _how(result: ExperimentResult, dataset: Dataset, info: ModelInfo) -> str:
    parts = [
        f"{dataset.splits[partition].X.shape[0]} no {partition.label.lower()}"
        for partition in (Partition.TRAIN, Partition.VALIDATION, Partition.TEST)
        if partition in dataset.splits
    ]
    lines = [
        f"- **Dados:** {result.dataset_name}, com {format_list(parts)} linhas (seed "
        f"{result.seed}).",
        "- **Preparação**, pela ordem:",
        *(
            f"  - {step.description}"
            for step in dataset.lineage
            if step.stage in _PREPARATION_STAGES
        ),
    ]
    settings = ", ".join(
        f"{key} = {format_value(value)}" for key, value in sorted(result.hyperparameters.items())
    )
    lines.append(
        f"- **Modelo:** {info.display_name} ({info.family.label.lower()}, "
        f"{info.implementation.label})" + (f", com {settings}." if settings else ".")
    )
    return "\n".join(lines)


def _performance(result: ExperimentResult) -> str:
    partitions = [partition for partition in Partition if partition in result.reports]
    header = "| Métrica | " + " | ".join(p.label for p in partitions) + " |"
    rule = "|---|" + "---|" * len(partitions)
    rows = []
    for metric in result.main_report.metrics:
        values = []
        for partition in partitions:
            found = result.reports[partition].find(metric.name)
            values.append("—" if found is None else found.formatted)
        rows.append(f"| {metric.display_name} | " + " | ".join(values) + " |")
    notes = [f"- {notice.message}" for notice in result.main_report.notices]
    return "\n".join([header, rule, *rows, *(["", *notes] if notes else [])])


def _limits(result: ExperimentResult, preparation: FittedPreparation, info: ModelInfo) -> str:
    lines = []
    for column in preparation.columns:
        if column.kind is ColumnKind.NUMERIC and column.minimum is not None:
            lines.append(
                f"- Viu {column.name} de {format_value(column.minimum)} a "
                f"{format_value(column.maximum)}: fora disso, a previsão é uma extrapolação."
            )
        elif column.kind is ColumnKind.CATEGORICAL and column.categories:
            shown = format_list(list(column.categories[:8]))
            more = "…" if len(column.categories) > 8 else ""
            lines.append(
                f"- {column.name} só com {shown}{more}: uma categoria nova conta como nenhuma."
            )
    if not info.supports_new_data:
        lines.append(f"- {info.display_name} não prevê linhas novas: só descreve as do treino.")
    if info.keeps_training_rows:
        lines.append(
            f"- {info.display_name} guarda linhas do treino dentro dele (já transformadas): "
            "quem tiver o modelo consegue vê-las."
        )
    lines.extend(
        f"- {notice.message}" for notice in result.notices if notice.level is not NoticeLevel.INFO
    )
    return "\n".join(lines) or "- O treino não deixou avisos."


def _files(files: Sequence[CardFile]) -> str:
    rows = [f"| `{file.path}` | {file.description} |" for file in files]
    return "\n".join(["| Ficheiro | O que tem |", "|---|---|", *rows])


def _usage(files: Sequence[CardFile], onnx_reason: str | None) -> str:
    paths = {file.path for file in files}
    lines = [
        '- **No ML Lab:** carrega este ficheiro no ecrã Guardar, em "Carregar um modelo '
        'treinado". O run entra no histórico e abre-se no Treinar e avaliar.',
    ]
    if "model.onnx" in paths:
        lines.extend(
            [
                "- **Noutra app:** o `model.onnx` junta a preparação e o modelo, e abre-se com o "
                "ONNX Runtime (em Python, C#, Java, JavaScript…). Cada coluna de `input.json` é "
                "uma entrada; o texto em falta vai como texto vazio. Em Python:",
                "",
                "  ```python",
                "  import onnxruntime as ort",
                "",
                '  session = ort.InferenceSession("model.onnx")',
                "  print([entrada.name for entrada in session.get_inputs()])",
                "  ```",
            ]
        )
    elif onnx_reason is not None:
        lines.append(f"- **Noutra app:** este pacote vai sem o `model.onnx`. {onnx_reason}")
    lines.append(
        "- **Os ficheiros:** o formato está descrito em `docs/formato-mllab.md`, no ML Lab. Os "
        "JSON leem-se em qualquer editor de texto."
    )
    if "model.joblib" not in paths:
        lines.append("- Este pacote vai **sem o modelo** (só a ficha): lê-se tudo, mas não prevê.")
    return "\n".join(lines)
