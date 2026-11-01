"""O ficheiro ONNX de um modelo: a preparação, o modelo, as saídas com nome e a confirmação.

O ficheiro recebe as colunas como chegam (uma entrada ``(N, 1)`` por coluna) e dá a previsão
com nomes fixos, para uma app saber o que pedir:

- classificação: ``label`` (a classe, pelo nome) e, se o modelo as der, ``probabilities``;
- regressão: ``prediction``; clustering: ``cluster``; anomalias: ``anomaly`` (1 ou 0);
- redução de dimensionalidade: ``projection``.

Antes de ir para um pacote, o ficheiro corre no ONNX Runtime com as linhas de exemplo, e tem de
dar as mesmas previsões do laboratório. Se não der (uma conversão com um erro, uma diferença de
precisão que muda uma classe), fica de fora, com a razão: um ``model.onnx`` que prevê outra coisa
é pior do que nenhum.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import onnx
import onnxruntime
from numpy.typing import NDArray

from ml_lab.core.data.profiling.type_inference import BOOLEAN_WORDS
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.serving import FittedPreparation
from ml_lab.core.export.graph import DOUBLE, INT64, STRING, Graph
from ml_lab.core.export.models import Kind, Prediction, kind_of, model_graph
from ml_lab.core.export.preparation import preparation_graph
from ml_lab.core.foundation.errors import CapabilityError
from ml_lab.core.models.base import LabModel

FORMAT: Final = "ml-lab-onnx"
"""O nome do formato, nos metadados do ficheiro."""

RELATIVE: Final = 1e-4
"""A diferença relativa que ainda conta como o mesmo número (os modelos que correm em
``float32``, como as árvores, afastam-se do laboratório no sétimo algarismo)."""

_TOO_DIFFERENT: Final = (
    "O ONNX não deu as mesmas previsões do laboratório nas linhas de exemplo, por isso fica de "
    "fora: um modelo que prevê outra coisa é pior do que nenhum."
)


@dataclass(frozen=True, slots=True)
class OnnxInput:
    """Uma entrada do ficheiro ONNX, com um valor típico (para o exemplo da ficha).

    Attributes:
        name: O nome da entrada (o da coluna).
        text: Se recebe texto (uma categoria) ou um número.
        typical: Um valor típico do treino, como o ONNX o recebe.
    """

    name: str
    text: bool
    typical: float | str


@dataclass(frozen=True, slots=True)
class OnnxSignature:
    """O que um ficheiro ONNX recebe e o que dá.

    Attributes:
        inputs: As entradas, uma por coluna.
        outputs: Os nomes das saídas, por ordem.
    """

    inputs: tuple[OnnxInput, ...]
    outputs: tuple[str, ...]


def export_onnx(model: LabModel[Any], preparation: FittedPreparation) -> bytes:
    """A preparação e o modelo num só ficheiro ONNX, confirmado nas linhas de exemplo.

    Raises:
        CapabilityError: Se a preparação ou o modelo não passarem para ONNX, ou se o ONNX não
            der as mesmas previsões do laboratório (a mensagem diz porquê, para a ficha).
    """
    graph = Graph()
    features = preparation_graph(graph, preparation)
    prediction = model_graph(graph, model, features, len(preparation.feature_names))
    kind = kind_of(model)
    outputs = _outputs(graph, kind, prediction, preparation.class_labels)
    classes = [preparation.class_labels[code] for code in prediction.classes]
    content = graph.model(
        doc=(
            f"{model.info.display_name}, com a preparação dos dados à frente (ML Lab). Uma "
            "entrada (N, 1) por coluna: os números em double, com NaN onde faltam; as "
            f"categorias em texto, vazio onde faltam. Saídas: {', '.join(outputs)}."
        ),
        metadata={
            "ml_lab.format": FORMAT,
            "ml_lab.model": model.info.name,
            "ml_lab.kind": kind,
            "ml_lab.target": preparation.target or "",
            "ml_lab.classes": json.dumps(classes, ensure_ascii=False),
            "ml_lab.features": json.dumps(list(preparation.feature_names), ensure_ascii=False),
        },
    ).SerializeToString()
    content = bytes(content)
    inputs, rows = preparation.check_rows()
    if not len(rows):
        raise CapabilityError(
            "Não há linhas de exemplo completas para confirmar a conversão, por isso o ONNX "
            "fica de fora."
        )
    _confirm(run_onnx(content, inputs), model, rows, preparation.class_labels)
    return content


def model_onnx(model: LabModel[Any], n_features: int, class_labels: Sequence[str] = ()) -> bytes:
    """Só o modelo, a partir das features já preparadas (a entrada ``features``, ``(N, F)``).

    Serve os testes de contrato de cada modelo, que treinam sem uma preparação.

    Raises:
        CapabilityError: Se o modelo não passar para ONNX.
    """
    graph = Graph()
    features = graph.input("features", DOUBLE, n_features)
    prediction = model_graph(graph, model, features, n_features)
    labels = tuple(class_labels) or tuple(
        str(code) for code in range(max(prediction.classes, default=-1) + 1)
    )
    _outputs(graph, kind_of(model), prediction, labels)
    return bytes(graph.model(doc=model.info.display_name, metadata={}).SerializeToString())


def preparation_onnx(preparation: FittedPreparation) -> bytes:
    """Só a preparação: as colunas como chegam, e as features (a saída ``features``, ``(N, F)``).

    Serve para ver (e testar) a preparação sozinha, transformação a transformação.

    Raises:
        CapabilityError: Se a preparação não passar para ONNX.
    """
    graph = Graph()
    features = preparation_graph(graph, preparation)
    graph.output(features, "features", DOUBLE, ["N", len(preparation.feature_names)])
    return bytes(
        graph.model(doc="A preparação dos dados (ML Lab).", metadata={}).SerializeToString()
    )


def run_onnx(content: bytes, inputs: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    """Corre um ficheiro ONNX no ONNX Runtime; devolve cada saída pelo nome.

    Args:
        content: O ficheiro.
        inputs: As entradas pelo nome (as que o ficheiro não pede ignoram-se).
    """
    session = onnxruntime.InferenceSession(content, providers=["CPUExecutionProvider"])
    feeds = {item.name: inputs[item.name] for item in session.get_inputs()}
    names = [item.name for item in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds), strict=True))


def onnx_signature(content: bytes, preparation: FittedPreparation) -> OnnxSignature:
    """As entradas (com um valor típico de cada) e as saídas de um ficheiro ONNX."""
    graph = onnx.load_from_string(content).graph
    columns = {column.name: column for column in preparation.columns}
    inputs = []
    for value in graph.input:
        column = columns[value.name]
        text = value.type.tensor_type.elem_type == STRING
        inputs.append(OnnxInput(value.name, text, _typical(column.typical, column.kind, text)))
    return OnnxSignature(tuple(inputs), tuple(value.name for value in graph.output))


def _outputs(
    graph: Graph, kind: Kind, prediction: Prediction, class_labels: Sequence[str]
) -> tuple[str, ...]:
    """As saídas com os nomes que as apps pedem; devolve-os, por ordem."""
    values = _flat(graph, prediction.values) if kind != "reduction" else prediction.values
    if kind == "classification":
        names = graph.constant(np.array(list(class_labels), dtype=object), "nomes_das_classes")
        codes = graph.op("Cast", [values], "codigos", to=INT64)
        graph.output(graph.op("Gather", [names, codes], "nome", axis=0), "label", STRING, ["N"])
        if prediction.probabilities is None:
            return ("label",)
        wide = graph.op("Cast", [prediction.probabilities], "probabilidades_double", to=DOUBLE)
        graph.output(wide, "probabilities", DOUBLE, ["N", len(prediction.classes)])
        return ("label", "probabilities")
    if kind in ("clustering", "anomaly"):
        name = "cluster" if kind == "clustering" else "anomaly"
        graph.output(graph.op("Cast", [values], f"{name}_int", to=INT64), name, INT64, ["N"])
        return (name,)
    name = "prediction" if kind == "regression" else "projection"
    shape: list[int | str] = ["N"] if kind == "regression" else ["N", "K"]
    graph.output(graph.op("Cast", [values], f"{name}_double", to=DOUBLE), name, DOUBLE, shape)
    return (name,)


def _confirm(
    outputs: Mapping[str, NDArray[Any]],
    model: LabModel[Any],
    rows: NDArray[np.float64],
    class_labels: Sequence[str],
) -> None:
    """Confirma que o ONNX deu as mesmas previsões do laboratório nas mesmas linhas."""
    kind = kind_of(model)
    if kind == "reduction":
        same = _close(outputs["projection"], model.transform(rows))
    elif kind == "classification":
        expected = np.asarray(model.predict(rows), dtype=np.int64)
        names = [class_labels[code] for code in expected]
        same = [str(label) for label in outputs["label"]] == names
        if same and "probabilities" in outputs:
            same = np.allclose(outputs["probabilities"], model.predict_proba(rows), atol=RELATIVE)
    elif kind == "regression":
        same = _close(outputs["prediction"], model.predict(rows))
    else:
        same = np.array_equal(
            outputs[kind if kind == "anomaly" else "cluster"], model.predict(rows)
        )
    if not same:
        raise CapabilityError(_TOO_DIFFERENT)


def _close(got: NDArray[Any], expected: NDArray[Any]) -> bool:
    """Se os números são os mesmos, a menos da precisão do ``float32``."""
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    return got.shape == expected.shape and bool(
        np.allclose(got, expected, rtol=RELATIVE, atol=1e-6 * scale)
    )


def _flat(graph: Graph, value: str) -> str:
    shape = graph.constant(np.array([-1], dtype=np.int64), "plano")
    return graph.op("Reshape", [value, shape], f"{value}_plano")


def _typical(value: Any, kind: ColumnKind, text: bool) -> float | str:
    """Um valor típico de uma coluna, como o ONNX o recebe (o texto como o one-hot o compara)."""
    if text:
        return "" if value is None else str(value)
    if kind is ColumnKind.BOOLEAN and isinstance(value, str):
        word = BOOLEAN_WORDS.get(value.strip().lower())
        return float("nan") if word is None else float(word)
    return float("nan") if value is None else float(value)
