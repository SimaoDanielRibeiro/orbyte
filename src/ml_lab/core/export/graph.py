"""Um construtor de grafos ONNX pequeno: nomes únicos, constantes, nós e grafos de fora.

Um modelo ONNX é um grafo: cada nó é uma operação (somar, comparar, multiplicar matrizes...)
que lê tensores com nome e escreve outros, e as constantes (os valores aprendidos no treino)
vão dentro do ficheiro. Quem o abre (o ONNX Runtime, noutra app, noutra linguagem) só precisa
de saber os nomes das entradas e das saídas.

Este construtor dá a cada tensor um nome que ainda não existe, guarda as constantes e junta
grafos feitos noutro lado (os do skl2onnx), com os nomes deles prefixados para não chocarem.
"""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, version_converter

OPSET: Final = 19
"""A versão das operações base. O ``Equal`` compara texto a partir da 19 (o one-hot precisa)."""

ML_OPSET: Final = 3
"""A versão das operações de machine learning (``ai.onnx.ml``: árvores, SVM, vizinhos...)."""

ML_DOMAIN: Final = "ai.onnx.ml"

IR_VERSION: Final = 9
"""A versão do formato do ficheiro, a que a opset 19 pede: os runtimes mais antigos leem-na."""

DOUBLE: Final = TensorProto.DOUBLE
FLOAT: Final = TensorProto.FLOAT
INT64: Final = TensorProto.INT64
STRING: Final = TensorProto.STRING

OUTPUT_NAMES: Final = ("label", "probabilities", "prediction", "cluster", "anomaly", "projection")
"""Os nomes das saídas que as apps pedem: ficam reservados, nenhum tensor interno os usa."""


class Graph:
    """Os nós, as constantes, as entradas e as saídas de um grafo a montar."""

    def __init__(self) -> None:
        self._nodes: list[onnx.NodeProto] = []
        self._initializers: list[onnx.TensorProto] = []
        self._inputs: list[onnx.ValueInfoProto] = []
        self._outputs: list[onnx.ValueInfoProto] = []
        self._used: set[str] = set(OUTPUT_NAMES)
        self._counter = itertools.count(1)
        self._opsets: dict[str, int] = {"": OPSET}

    def fresh(self, hint: str) -> str:
        """Um nome de tensor que ainda não existe, a partir de ``hint``."""
        name = hint
        while name in self._used:
            name = f"{hint}_{next(self._counter)}"
        self._used.add(name)
        return name

    def input(self, name: str, elem_type: int, columns: int) -> str:
        """Uma entrada do grafo, ``(linhas, columns)``, com o nome exato que as apps usam."""
        if name in self._used:
            raise ValueError(f"The name {name!r} already exists in the graph.")
        self._used.add(name)
        self._inputs.append(helper.make_tensor_value_info(name, elem_type, ["N", columns]))
        return name

    def constant(self, value: Any, hint: str, *, dtype: Any = None) -> str:
        """Uma constante do grafo (um valor aprendido): números, ou texto."""
        name = self.fresh(hint)
        array = np.asarray(value, dtype=dtype)
        if array.dtype.kind in "OUS":
            texts = np.array([str(item).encode("utf-8") for item in array.ravel()], dtype=object)
            tensor = helper.make_tensor(name, STRING, list(array.shape), texts)
        else:
            tensor = numpy_helper.from_array(array, name)
        self._initializers.append(tensor)
        return name

    def op(
        self, op_type: str, inputs: Sequence[str], hint: str, *, domain: str = "", **attributes: Any
    ) -> str:
        """Um nó com uma só saída; devolve o nome dela."""
        return self.ops(op_type, inputs, [hint], domain=domain, **attributes)[0]

    def ops(
        self,
        op_type: str,
        inputs: Sequence[str],
        hints: Sequence[str],
        *,
        domain: str = "",
        **attributes: Any,
    ) -> list[str]:
        """Um nó com várias saídas (por exemplo, o ``TopK``: os valores e as posições)."""
        outputs = [self.fresh(hint) for hint in hints]
        node = helper.make_node(
            op_type, list(inputs), outputs, name=f"{op_type}:{outputs[0]}", domain=domain
        )
        node.attribute.extend(
            helper.make_attribute(key, value) for key, value in attributes.items()
        )
        self._nodes.append(node)
        return outputs

    def output(self, value: str, name: str, elem_type: int, shape: Sequence[int | str]) -> None:
        """Uma saída do grafo, com o nome exato que as apps pedem."""
        if name not in OUTPUT_NAMES and name in self._used:
            raise ValueError(f"The name {name!r} already exists in the graph.")
        self._used.add(name)
        self._nodes.append(helper.make_node("Identity", [value], [name], name=f"saida:{name}"))
        self._outputs.append(helper.make_tensor_value_info(name, elem_type, list(shape)))

    def inline(
        self, model: onnx.ModelProto, feeds: Mapping[str, str], prefix: str
    ) -> dict[str, str]:
        """Junta ao grafo o grafo de ``model`` (por exemplo, um do skl2onnx).

        Os nomes dele ganham ``prefix``, e as operações base passam para a versão deste grafo.

        Args:
            model: O modelo a juntar.
            feeds: O tensor deste grafo que alimenta cada entrada dele.
            prefix: O começo dos nomes dele.

        Returns:
            O nome que cada saída dele ficou a ter neste grafo.
        """
        versions = {opset.domain: opset.version for opset in model.opset_import}
        if versions.get("", OPSET) != OPSET:
            model = version_converter.convert_version(model, OPSET)
        for domain, version in versions.items():
            if domain:
                self._opsets[domain] = max(self._opsets.get(domain, 0), version)
        prefixed = onnx.compose.add_prefix(model, prefix)
        graph = prefixed.graph
        self._used.update(
            {name for node in graph.node for name in [*node.input, *node.output] if name},
        )
        self._used.update(tensor.name for tensor in graph.initializer)
        for value in graph.input:
            self._used.add(value.name)
            source = feeds[value.name.removeprefix(prefix)]
            self._nodes.append(
                helper.make_node("Identity", [source], [value.name], name=f"liga:{value.name}")
            )
        self._nodes.extend(graph.node)
        self._initializers.extend(graph.initializer)
        return {value.name.removeprefix(prefix): value.name for value in graph.output}

    def model(self, *, doc: str, metadata: Mapping[str, str]) -> onnx.ModelProto:
        """O modelo ONNX, confirmado pelo verificador do ``onnx``."""
        graph = helper.make_graph(
            self._nodes, "ml_lab", self._inputs, self._outputs, self._initializers
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid(domain, version) for domain, version in self._opsets.items()
            ],
            producer_name="Orbyte",
            doc_string=doc,
        )
        model.ir_version = IR_VERSION
        helper.set_model_props(model, dict(metadata))
        onnx.checker.check_model(model, full_check=True)
        return model
