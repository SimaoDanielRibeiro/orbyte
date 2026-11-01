"""A preparação em ONNX: as colunas como chegam, e as features que o modelo conhece.

Uma app de fora não tem o pandas nem o laboratório: recebe uma linha com ``Sex = "female"`` e
``Age = 22``, e o modelo precisa de ``Sex_female = 1`` e da idade padronizada. Por isso a
preparação vai dentro do ficheiro, à frente do modelo, com os valores aprendidos no treino como
constantes. Cada transformação vira umas poucas operações:

- **limitar um extremo** (a limpeza que vale para linhas novas): ``Clip``, com os limites;
- **imputar**: "se falta, o valor aprendido" (``Where``, com ``IsNaN`` nos números e o texto
  vazio nas categorias);
- **intervalos**: quantos limites do treino o valor já passou (``GreaterOrEqual`` e uma soma);
- **one-hot**: uma comparação por categoria (``Equal``), que dá 1 ou 0;
- **polinomiais**: produtos (``Mul``);
- **escalar**: ``(x - média) / desvio``, ou ``(x - mínimo) / amplitude``.

No fim, as features juntam-se pela ordem que o modelo conhece (``Concat``), em ``double``: as
contas são as mesmas do laboratório, com a mesma precisão.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_lab.core.data.cleaning.base import AppliedStep
from ml_lab.core.data.profiling.type_inference import BOOLEAN_WORDS
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.serving import FittedPreparation
from ml_lab.core.data.transforms.base import DropColumns, Transform
from ml_lab.core.data.transforms.encoding import BooleanEncoder, OneHotEncoder
from ml_lab.core.data.transforms.feature_engineering import (
    Binning,
    DateFeatures,
    PolynomialFeatures,
)
from ml_lab.core.data.transforms.imputation import Imputer
from ml_lab.core.data.transforms.scaling import MinMaxScaler, StandardScaler
from ml_lab.core.export.graph import DOUBLE, STRING, Graph
from ml_lab.core.foundation.errors import CapabilityError

_REPLAYS_WITHOUT_NODES = frozenset({"drop_columns", "convert_type"})
"""As limpezas de linhas novas que não precisam de nós: as colunas removidas não são entradas, e
as entradas já chegam com o tipo certo (os números como números)."""


@dataclass(slots=True)
class _Column:
    """Uma coluna a meio da preparação: o tensor ``(N, 1)`` que a tem e o que ela é."""

    value: str
    kind: ColumnKind

    @property
    def text(self) -> bool:
        return self.kind is ColumnKind.CATEGORICAL


def preparation_graph(graph: Graph, preparation: FittedPreparation) -> str:
    """Junta ao grafo uma entrada por coluna e a preparação; devolve as features ``(N, F)``.

    As colunas numéricas e booleanas entram como ``double`` (``NaN`` quando faltam; os
    booleanos como 0 ou 1); as categóricas, como texto (vazio quando faltam).

    Raises:
        CapabilityError: Se a preparação tiver uma parte que ainda não passa para ONNX (as
            datas, por exemplo).
    """
    frame = _inputs(graph, preparation)
    for applied in preparation.cleaning:
        _clean(graph, frame, applied)
    for step in preparation.pipeline.steps:
        _transform(graph, frame, step)
    features = []
    for name in preparation.feature_names:
        column = frame.get(name)
        if column is None or column.kind not in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN):
            raise CapabilityError(
                f"A feature {name} não sai da preparação como um número: esta preparação "
                "ainda não passa para ONNX."
            )
        features.append(column.value)
    return graph.op("Concat", features, "features_preparadas", axis=1)


def _inputs(graph: Graph, preparation: FittedPreparation) -> dict[str, _Column]:
    """As entradas: uma por coluna que o modelo espera, como chega."""
    frame: dict[str, _Column] = {}
    for column in preparation.columns:
        if column.kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN):
            frame[column.name] = _Column(graph.input(column.name, DOUBLE, 1), column.kind)
        elif column.kind is ColumnKind.CATEGORICAL:
            frame[column.name] = _Column(graph.input(column.name, STRING, 1), column.kind)
        else:
            # Uma data (ou texto livre) só pode sair pela preparação; se for usada, falha lá.
            frame[column.name] = _Column("", column.kind)
    return frame


def _clean(graph: Graph, frame: dict[str, _Column], applied: AppliedStep) -> None:
    """A limpeza que se repete nas linhas novas (limitar os extremos, com os limites aprendidos)."""
    if not applied.replayable or applied.step.operation in _REPLAYS_WITHOUT_NODES:
        return
    params, learned = applied.step.params, applied.learned
    if applied.step.operation != "treat_outliers":
        raise CapabilityError(f"A limpeza {applied.step.operation} ainda não passa para ONNX.")
    column = frame.get(str(params["column"]))
    if params.get("method", "clip") != "clip" or "lower" not in learned or column is None:
        return
    value = _number(column, "limitar os extremos")
    lower = graph.constant(np.float64(learned["lower"]), "limite_inferior")
    upper = graph.constant(np.float64(learned["upper"]), "limite_superior")
    clipped = graph.op("Clip", [value, lower, upper], f"{params['column']}_limitada")
    # Um valor em falta continua em falta (como no pandas), em vez de virar um dos limites.
    missing = graph.op("IsNaN", [value], f"{params['column']}_falta")
    column.value = graph.op("Where", [missing, value, clipped], f"{params['column']}_limpa")


def _transform(graph: Graph, frame: dict[str, _Column], step: Transform) -> None:
    """Uma transformação aprendida, sobre as colunas a meio da preparação."""
    match step:
        case DropColumns():
            for name in step.columns:
                frame.pop(name, None)
        case DateFeatures():
            raise CapabilityError(
                "As datas ainda não passam para ONNX: o formato não tem um tipo de data, e as "
                "partes (ano, mês, dia) teriam de ser calculadas pela app."
            )
        case Imputer():
            for name in step.columns:
                _impute(graph, frame[name], name, step.fill_values[name])
        case Binning():
            for name in step.columns:
                column = frame.pop(name)
                frame[Binning.output_name(name)] = _Column(
                    _bins(graph, _number(column, "dividir em intervalos"), name, step.edges[name]),
                    ColumnKind.NUMERIC,
                )
        case BooleanEncoder():
            pass  # os booleanos já entram como 0 e 1
        case OneHotEncoder():
            for name in step.columns:
                _one_hot(graph, frame, name, step.categories[name])
        case PolynomialFeatures():
            for term, output in zip(step.terms(), step.output_names(), strict=True):
                factors = [_number(frame[name], "multiplicar") for name in term]
                product = factors[0]
                for factor in factors[1:]:
                    product = graph.op("Mul", [product, factor], output)
                frame[output] = _Column(product, ColumnKind.NUMERIC)
        case StandardScaler():
            for name in step.columns:
                _scale(graph, frame[name], name, step.means[name], step.scales[name])
        case MinMaxScaler():
            for name in step.columns:
                _scale(graph, frame[name], name, step.minimums[name], step.ranges[name])
        case _:
            raise CapabilityError(f"A transformação «{step.title}» ainda não passa para ONNX.")


def _impute(graph: Graph, column: _Column, name: str, fill: Any) -> None:
    """``Where(falta, valor aprendido, x)``: nos números falta o ``NaN``; no texto, o vazio."""
    if column.text:
        empty = graph.constant(np.array([""], dtype=object), "vazio")
        missing = graph.op("Equal", [column.value, empty], f"{name}_falta")
        value = graph.constant(np.array([str(fill)], dtype=object), f"{name}_preenche")
    else:
        number = _zero_one(fill) if column.kind is ColumnKind.BOOLEAN else float(fill)
        missing = graph.op("IsNaN", [_number(column, "imputar")], f"{name}_falta")
        value = graph.constant(np.array([number], dtype=np.float64), f"{name}_preenche")
    column.value = graph.op("Where", [missing, value, column.value], f"{name}_imputada")


def _bins(graph: Graph, value: str, name: str, edges: tuple[float, ...]) -> str:
    """O número do intervalo: quantos limites interiores o valor já passou (o ``digitize``)."""
    inner = np.asarray(edges[1:-1], dtype=np.float64).reshape(1, -1)
    missing = graph.op("IsNaN", [value], f"{name}_falta")
    if inner.size == 0:
        zero = graph.constant(np.zeros((1, 1)), f"{name}_um_intervalo")
        return graph.op("Where", [missing, value, zero], f"{name}_bin")
    passed = graph.op(
        "GreaterOrEqual", [value, graph.constant(inner, f"{name}_limites")], f"{name}_passou"
    )
    counted = graph.op("Cast", [passed], f"{name}_passou_num", to=DOUBLE)
    axes = graph.constant(np.array([1], dtype=np.int64), "eixo_1")
    bins = graph.op("ReduceSum", [counted, axes], f"{name}_intervalo", keepdims=1)
    return graph.op("Where", [missing, value, bins], f"{name}_bin")


def _one_hot(
    graph: Graph, frame: dict[str, _Column], name: str, categories: tuple[str, ...]
) -> None:
    """Uma coluna 0/1 por categoria do treino; uma categoria nova fica com zeros em todas."""
    column = frame.pop(name)
    if not column.text:
        raise CapabilityError(
            f"A coluna {name} não é de texto: o one-hot dela não passa para ONNX."
        )
    for category in categories:
        constant = graph.constant(np.array([category], dtype=object), f"{name}_{category}_texto")
        same = graph.op("Equal", [column.value, constant], f"{name}_{category}_igual")
        frame[f"{name}_{category}"] = _Column(
            graph.op("Cast", [same], f"{name}_{category}", to=DOUBLE), ColumnKind.NUMERIC
        )


def _scale(graph: Graph, column: _Column, name: str, center: float, scale: float) -> None:
    """``(x - centro) / escala``, com os valores do treino."""
    value = _number(column, "escalar")
    shifted = graph.op(
        "Sub", [value, graph.constant(np.float64(center), f"{name}_centro")], f"{name}_centrada"
    )
    column.value = graph.op(
        "Div", [shifted, graph.constant(np.float64(scale), f"{name}_escala")], f"{name}_escalada"
    )


def _number(column: _Column, doing: str) -> str:
    """O tensor de uma coluna numérica, ou a razão por que não se pode ``doing``."""
    if column.kind not in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN) or not column.value:
        raise CapabilityError(
            f"Para {doing}, a coluna tinha de ser um número, e é {column.kind.label.lower()}: "
            "esta preparação ainda não passa para ONNX."
        )
    return column.value


def _zero_one(value: Any) -> float:
    """O valor de preenchimento de uma coluna booleana, em 0 ou 1 (como entra no ONNX)."""
    if isinstance(value, str):
        word = BOOLEAN_WORDS.get(value.strip().lower())
        return float("nan") if word is None else float(word)
    return float(value)
