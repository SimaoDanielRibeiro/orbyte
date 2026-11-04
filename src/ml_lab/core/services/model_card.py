"""A ficha de um modelo: o ``README.md`` de um pacote, escrito para pessoas (ADR 0039).

Uma ficha de modelo diz o que o modelo prevê, com que dados aprendeu, como se portou, onde não
se deve confiar nele e como se usa. É o que quem recebe o ficheiro lê primeiro, antes de
abrir os JSON. Está em inglês e os números escrevem-se como em todo o site (ADR 0043).
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.lineage import Stage
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.serving import FittedPreparation
from ml_lab.core.export import OnnxSignature
from ml_lab.core.foundation.notices import NoticeLevel
from ml_lab.core.foundation.types import Partition
from ml_lab.core.models.base import ModelInfo
from ml_lab.core.services.experiments import ExperimentResult
from ml_lab.formatting import format_list, format_value

MAX_EXAMPLE_INPUTS: Final = 30
"""Quantas colunas o exemplo do ONNX escreve, no máximo (as outras ficam num comentário)."""

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
    onnx: OnnxSignature | None,
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
        onnx: As entradas e as saídas do ``model.onnx``, se ele vai no pacote (o exemplo da
            ficha usa-as).
        onnx_reason: Porque é que o pacote vai sem o ``model.onnx``, quando leva o modelo e
            não leva o ONNX (``None`` nos outros casos).
    """
    sections = [
        f"# {name}",
        _summary(result, preparation, info),
        "## How it was made",
        _how(result, dataset, info),
        "## How it did",
        _performance(result),
        "## Its limits",
        _limits(result, preparation, info),
        "## What's in this package",
        _files(files),
        "## How to use it",
        _usage(files, onnx, onnx_reason),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _summary(result: ExperimentResult, preparation: FittedPreparation, info: ModelInfo) -> str:
    columns = format_list([column.name for column in preparation.columns])
    task = result.task_type.label.lower()
    if preparation.target is not None and result.task_type.requires_target:
        what = f"predicts **{preparation.target}** ({task}) from {columns}"
    else:
        what = f"does {task} with {columns}"
    # Travessões, e não parênteses: o nome do modelo pode ter os seus (o gradiente descendente).
    return f"This model — {info.display_name} — {what}. It was trained in Orbyte."


def _how(result: ExperimentResult, dataset: Dataset, info: ModelInfo) -> str:
    parts = [
        f"{dataset.splits[partition].X.shape[0]} for {partition.label.lower()}"
        for partition in (Partition.TRAIN, Partition.VALIDATION, Partition.TEST)
        if partition in dataset.splits
    ]
    parts[0] = parts[0].replace(" for ", " rows for ", 1)
    lines = [
        f"- **Data:** {result.dataset_name}: {format_list(parts)} (seed {result.seed}).",
        "- **Preparation**, in order:",
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
        f"- **Model:** {info.display_name} ({info.family.label.lower()}, "
        f"{info.implementation.label})" + (f", with {settings}." if settings else ".")
    )
    return "\n".join(lines)


def _performance(result: ExperimentResult) -> str:
    partitions = [partition for partition in Partition if partition in result.reports]
    header = "| Metric | " + " | ".join(p.label for p in partitions) + " |"
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
                f"- It saw {column.name} from {format_value(column.minimum)} to "
                f"{format_value(column.maximum)}: outside that, the prediction is an "
                "extrapolation."
            )
        elif column.kind is ColumnKind.CATEGORICAL and column.categories:
            shown = format_list(list(column.categories[:8]))
            more = "…" if len(column.categories) > 8 else ""
            lines.append(f"- {column.name} only with {shown}{more}: a new category counts as none.")
    if not info.supports_new_data:
        lines.append(
            f"- {info.display_name} doesn't predict new rows: it only describes the training ones."
        )
    if info.keeps_training_rows:
        lines.append(
            f"- {info.display_name} keeps training rows inside it (already transformed): "
            "whoever has the model can see them."
        )
    lines.extend(
        f"- {notice.message}" for notice in result.notices if notice.level is not NoticeLevel.INFO
    )
    return "\n".join(lines) or "- The training left no warnings."


def _onnx_example(onnx: OnnxSignature) -> list[str]:
    """Um exemplo em Python que corre o ``model.onnx`` com uma linha de valores típicos."""
    feeds = [
        f'    "{entry.name}": np.array([[{_literal(entry.typical)}]]'
        + (", dtype=object)," if entry.text else "),")
        for entry in onnx.inputs[:MAX_EXAMPLE_INPUTS]
    ]
    if len(onnx.inputs) > MAX_EXAMPLE_INPUTS:
        feeds.append(f"    # … and the other {len(onnx.inputs) - MAX_EXAMPLE_INPUTS} columns")
    outputs = ", ".join(onnx.outputs)
    received = f"{outputs} = " if len(onnx.outputs) > 1 else f"({outputs},) = "
    return [
        "```python",
        "import numpy as np",
        "import onnxruntime as ort",
        "",
        'session = ort.InferenceSession("model.onnx")',
        "row = {",
        *feeds,
        "}",
        f"{received}session.run(None, row)",
        f"print({outputs})",
        "```",
    ]


def _literal(value: float | str) -> str:
    """O valor escrito em Python: o texto entre aspas, o número tal como é (``nan`` à parte)."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return "np.nan" if math.isnan(value) else repr(float(value))


def _files(files: Sequence[CardFile]) -> str:
    rows = [f"| `{file.path}` | {file.description} |" for file in files]
    return "\n".join(["| File | What it has |", "|---|---|", *rows])


def _usage(files: Sequence[CardFile], onnx: OnnxSignature | None, onnx_reason: str | None) -> str:
    paths = {file.path for file in files}
    lines = [
        '- **In Orbyte:** load this file in the Save screen, under "Load a trained model". The '
        "run goes into the history and opens in Train & evaluate.",
    ]
    if onnx is not None:
        lines.extend(
            [
                "- **In another app:** `model.onnx` joins the preparation and the model, and opens "
                "with ONNX Runtime (in Python, C#, Java, JavaScript…), without Orbyte. Each column "
                "is an input with one row per example: the numbers as `float64` (`nan` when "
                "missing) and the categories as text (empty when missing). In Python, with the "
                "typical training values:",
                "",
                *(f"  {line}" if line else "" for line in _onnx_example(onnx)),
            ]
        )
    elif onnx_reason is not None:
        lines.append(
            f"- **In another app:** this package comes without `model.onnx`. {onnx_reason}"
        )
    lines.append(
        "- **The files:** the format is described in `docs/formato-mllab.md`, in the Orbyte "
        "repository. The JSON files can be read in any text editor."
    )
    if "model.joblib" not in paths:
        lines.append(
            "- This package comes **without the model** (only the card): everything can be read, "
            "but it doesn't predict."
        )
    return "\n".join(lines)
