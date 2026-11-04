"""Os modelos em ONNX: um conversor por modelo, num registo.

- **Os do scikit-learn** passam pelo skl2onnx, o conversor oficial. Primeiro em ``double``
  (as contas ficam iguais às do laboratório); se o skl2onnx ou o ONNX Runtime não o aceitarem
  (as árvores, por exemplo), em ``float``, que é como as árvores do scikit-learn já comparam.
- **Os didáticos e as redes** montam-se a partir do que aprenderam: os pesos, os centróides, as
  componentes, as linhas que memorizam. As contas são as mesmas do NumPy (a mesma fórmula da
  distância, a mesma sigmoide), para darem as mesmas previsões.
- **O t-SNE** não tem conversor: não sabe colocar pontos novos.

Cada conversor devolve a previsão do modelo no grafo (``Prediction``); as saídas com os nomes
que as apps pedem (``label``, ``prediction``...) montam-se depois, iguais para todos.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import onnx
import onnxruntime
from numpy.typing import NDArray
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

from ml_lab.core.export.graph import DOUBLE, FLOAT, INT64, ML_DOMAIN, ML_OPSET, OPSET, Graph
from ml_lab.core.foundation.errors import CapabilityError
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import LabModel
from ml_lab.core.models.parameters import DenseLayer, Memory

type Kind = Literal["classification", "regression", "clustering", "anomaly", "reduction"]


@dataclass(frozen=True, slots=True)
class Prediction:
    """A previsão de um modelo no grafo, antes de levar o nome que as apps pedem.

    Attributes:
        values: O tensor da previsão: na classificação, a classe (o código, ``int64``); na
            regressão, o valor; no clustering, o grupo; na deteção de anomalias, 1 ou 0; na
            redução de dimensionalidade, a projeção ``(N, k)``.
        probabilities: As probabilidades de cada classe, se o modelo as der.
        classes: As classes das colunas de ``probabilities``, por ordem (os códigos).
    """

    values: str
    probabilities: str | None = None
    classes: tuple[int, ...] = ()


type Converter = Callable[[Graph, LabModel[Any], str, int], Prediction]
"""Um conversor: o grafo, o modelo treinado, as features ``(N, F)`` em ``double`` e ``F``."""

CONVERTERS: Registry[Converter] = Registry("onnx_converters")
"""Os conversores próprios, pelo nome do modelo. Os do scikit-learn sem conversor próprio
passam pelo skl2onnx."""


@runtime_checkable
class _Layered(Protocol):
    def dense_layers(self) -> list[DenseLayer]: ...


@runtime_checkable
class _Remembering(Protocol):
    def memory(self) -> Memory: ...


def kind_of(model: LabModel[Any]) -> Kind:
    """O que o modelo dá: uma classe, um número, um grupo, uma marca de anomalia ou uma projeção."""
    tasks = model.info.task_types
    if any(task.is_classification for task in tasks):
        return "classification"
    if TaskType.REGRESSION in tasks:
        return "regression"
    if TaskType.CLUSTERING in tasks:
        return "clustering"
    if TaskType.ANOMALY_DETECTION in tasks:
        return "anomaly"
    return "reduction"


def model_graph(graph: Graph, model: LabModel[Any], features: str, n_features: int) -> Prediction:
    """Junta ao grafo o modelo, a partir das features ``(N, F)`` em ``double``.

    Raises:
        CapabilityError: Se o modelo não souber prever linhas novas, ou ainda não tiver
            conversão para ONNX.
    """
    info = model.info
    if not info.supports_new_data:
        raise CapabilityError(
            f"{info.display_name} can't place new points: there's nothing to take to another app."
        )
    if info.name in CONVERTERS:
        return CONVERTERS.get(info.name)(graph, model, features, n_features)
    if isinstance(model, SklearnModel):
        return _sklearn(graph, model, features, n_features)
    raise CapabilityError(f"{info.display_name} has no ONNX conversion yet.")


# --- scikit-learn ------------------------------------------------------------------------


def _sklearn(graph: Graph, model: SklearnModel[Any], features: str, n_features: int) -> Prediction:
    """Pelo skl2onnx: em ``double`` se der, senão em ``float``."""
    kind = kind_of(model)
    estimator = model.estimator_
    converted, elem_type = _converted(estimator, n_features, kind == "classification")
    feed = features
    if elem_type == FLOAT:
        feed = graph.op("Cast", [features], "features_float", to=FLOAT)
    outputs = graph.inline(converted, {"X": feed}, "sk_")
    if kind == "classification":
        probabilities = outputs.get("probabilities") if model.info.supports_proba else None
        classes = tuple(int(value) for value in estimator.classes_)
        return Prediction(outputs["label"], probabilities, classes)
    if kind == "anomaly":
        # O scikit-learn marca as anomalias com -1; o laboratório, com 1.
        minus_one = graph.constant(np.array([-1], dtype=np.int64), "menos_um")
        flagged = graph.op("Equal", [outputs["label"], minus_one], "e_anomalia")
        return Prediction(graph.op("Cast", [flagged], "anomalia", to=INT64))
    if kind == "clustering":
        return Prediction(outputs["label"])
    return Prediction(outputs["variable"])


def _converted(estimator: Any, n_features: int, classifier: bool) -> tuple[onnx.ModelProto, int]:
    """O estimador convertido, e o tipo das features que ele recebe (``double`` ou ``float``)."""
    problem = ""
    for elem_type, tensor_type in ((DOUBLE, DoubleTensorType), (FLOAT, FloatTensorType)):
        try:
            with warnings.catch_warnings():
                # O conversor do SVC no skl2onnx ainda lê probA_ e probB_, que o scikit-learn
                # 1.9 deu como obsoletos (o nosso SVM calibra as probabilidades à parte e não os
                # usa). O aviso é do skl2onnx, não do laboratório; a confirmação das previsões,
                # no fim da exportação, garante que a conversão está certa.
                warnings.filterwarnings(
                    "ignore",
                    message=r"Attribute `prob[AB]_` was deprecated",
                    category=FutureWarning,
                )
                converted = convert_sklearn(
                    estimator,
                    initial_types=[("X", tensor_type([None, n_features]))],
                    target_opset={"": OPSET, ML_DOMAIN: ML_OPSET},
                    options={id(estimator): {"zipmap": False}} if classifier else None,
                )
            _runs(converted, elem_type, n_features)
        # O skl2onnx e o ONNX Runtime falham com erros de muitos tipos (um conversor que
        # falta, um tipo que um nó não aceita): qualquer um quer dizer "assim não dá".
        except Exception as error:
            problem = str(error).splitlines()[0][:200]
            continue
        return converted, elem_type
    raise CapabilityError(f"skl2onnx doesn't convert {type(estimator).__name__} ({problem}).")


def _runs(model: onnx.ModelProto, elem_type: int, n_features: int) -> None:
    """Confirma que o ONNX Runtime abre e corre o modelo (há conversões que só falham aqui)."""
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    dtype = np.float64 if elem_type == DOUBLE else np.float32
    session.run(None, {"X": np.zeros((2, n_features), dtype=dtype)})


# --- os didáticos ------------------------------------------------------------------------


@CONVERTERS.register("linear_regression_gd")
def _linear_regression(
    graph: Graph, model: LabModel[Any], features: str, n_features: int
) -> Prediction:
    """``x · pesos + ordenada``."""
    state = model.state()
    weights = np.asarray(state["coefficients"], dtype=np.float64).reshape(-1, 1)
    product = graph.op("MatMul", [features, graph.constant(weights, "pesos")], "soma")
    intercept = graph.constant(np.float64(state["intercept"]), "ordenada")
    return Prediction(graph.op("Add", [product, intercept], "previsao"))


@CONVERTERS.register("logistic_regression_gd")
def _logistic_regression(
    graph: Graph, model: LabModel[Any], features: str, n_features: int
) -> Prediction:
    """A sigmoide (duas classes) ou o softmax das pontuações; ganha a classe mais provável."""
    scores, classes = _linear_scores(graph, model, features)
    if len(classes) == 2:
        # 0,5·(1 + tanh(z/2)): a sigmoide como o laboratório a calcula.
        half = graph.constant(np.float64(0.5), "meio")
        one = graph.constant(np.float64(1.0), "um")
        tanh = graph.op("Tanh", [graph.op("Mul", [half, scores], "meia_pontuacao")], "tanh")
        positive = graph.op("Mul", [half, graph.op("Add", [one, tanh], "um_mais_tanh")], "p")
        chosen = graph.op("GreaterOrEqual", [positive, half], "positiva")
        index = _flat(graph, graph.op("Cast", [chosen], "indice", to=INT64))
        negative = graph.op("Sub", [one, positive], "q")
        probabilities = graph.op("Concat", [negative, positive], "probabilidades", axis=1)
    else:
        probabilities = graph.op("Softmax", [scores], "probabilidades", axis=1)
        index = graph.op("ArgMax", [probabilities], "indice", axis=1, keepdims=0)
    codes = graph.constant(classes, "classes")
    label = graph.op("Gather", [codes, index], "classe", axis=0)
    return Prediction(label, probabilities, tuple(int(value) for value in classes))


@CONVERTERS.register("perceptron")
def _perceptron(graph: Graph, model: LabModel[Any], features: str, n_features: int) -> Prediction:
    """O sinal da pontuação (duas classes), ou a pontuação mais alta."""
    scores, classes = _linear_scores(graph, model, features)
    if len(classes) == 2:
        zero = graph.constant(np.float64(0.0), "zero")
        positive = graph.op("Greater", [scores, zero], "positiva")
        index = _flat(graph, graph.op("Cast", [positive], "indice", to=INT64))
    else:
        index = graph.op("ArgMax", [scores], "indice", axis=1, keepdims=0)
    codes = graph.constant(classes, "classes")
    label = graph.op("Gather", [codes, index], "classe", axis=0)
    return Prediction(label, None, tuple(int(value) for value in classes))


@CONVERTERS.register("kmeans_numpy")
def _kmeans(graph: Graph, model: LabModel[Any], features: str, n_features: int) -> Prediction:
    """O centróide mais próximo."""
    centers = np.asarray(model.state()["centers"], dtype=np.float64)
    distances = _euclidean(graph, features, centers, "centroides")
    return Prediction(graph.op("ArgMin", [distances], "grupo", axis=1, keepdims=0))


@CONVERTERS.register("pca_numpy")
def _pca(graph: Graph, model: LabModel[Any], features: str, n_features: int) -> Prediction:
    """``(x - média) · componentesᵀ``."""
    state = model.state()
    mean = graph.constant(np.asarray(state["mean"], dtype=np.float64), "media")
    components = np.asarray(state["components"], dtype=np.float64)
    centered = graph.op("Sub", [features, mean], "centradas")
    loadings = graph.constant(components.T.copy(), "componentes")
    return Prediction(graph.op("MatMul", [centered, loadings], "projecao"))


@CONVERTERS.register("knn_numpy")
def _knn(graph: Graph, model: LabModel[Any], features: str, n_features: int) -> Prediction:
    """Os k vizinhos mais próximos votam (com o mesmo peso, ou com o inverso da distância)."""
    memory = _memory(model)
    params = model.hyperparameters
    classes = np.unique(memory.labels)
    codes = np.searchsorted(classes, memory.labels).astype(np.int64)
    k = int(params.n_neighbors)
    if params.metric == "manhattan":
        distances = _manhattan(graph, features, memory.rows)
    else:
        distances = _euclidean(graph, features, memory.rows, "treino")
    negative = graph.op("Neg", [distances], "menos_distancias")
    count = graph.constant(np.array([k], dtype=np.int64), "k")
    _, nearest = graph.ops("TopK", [negative, count], ["perto", "vizinhos"], axis=1)
    near = graph.op("GatherElements", [distances, nearest], "distancias_vizinhos", axis=1)
    weights = _vote_weights(graph, near, distance=params.weights == "distance")
    neighbor_codes = graph.op("Gather", [graph.constant(codes, "codigos"), nearest], "cods")
    # O voto de cada vizinho, (N, k, classes): 1 na classe dele. Uma comparação com as classes,
    # e não o OneHot, que o ONNX Runtime não tem para todos os tipos.
    axes2 = graph.constant(np.array([2], dtype=np.int64), "eixo_2")
    column = graph.op("Unsqueeze", [neighbor_codes, axes2], "cods_3d")
    every = graph.constant(np.arange(len(classes), dtype=np.int64), "todas_as_classes")
    same = graph.op("Equal", [column, every], "e_desta_classe")
    one_hot = graph.op("Cast", [same], "voto_classe", to=DOUBLE)
    weighted = graph.op(
        "Mul", [one_hot, graph.op("Unsqueeze", [weights, axes2], "pesos_3d")], "votos_pesados"
    )
    axes1 = graph.constant(np.array([1], dtype=np.int64), "eixo_1")
    votes = graph.op("ReduceSum", [weighted, axes1], "votos", keepdims=0)
    total = graph.op("ReduceSum", [votes, axes1], "total_votos", keepdims=1)
    probabilities = graph.op("Div", [votes, total], "probabilidades")
    index = graph.op("ArgMax", [votes], "indice", axis=1, keepdims=0)
    label = graph.op("Gather", [graph.constant(classes, "classes"), index], "classe", axis=0)
    return Prediction(label, probabilities, tuple(int(value) for value in classes))


@CONVERTERS.register("agglomerative")
def _agglomerative(
    graph: Graph, model: LabModel[Any], features: str, n_features: int
) -> Prediction:
    """O grupo do ponto de treino mais próximo."""
    memory = _memory(model)
    distances = _euclidean(graph, features, memory.rows, "treino")
    closest = graph.op("ArgMin", [distances], "mais_proximo", axis=1, keepdims=0)
    labels = graph.constant(memory.labels.astype(np.int64), "grupos_treino")
    return Prediction(graph.op("Gather", [labels, closest], "grupo", axis=0))


@CONVERTERS.register("dbscan")
def _dbscan(graph: Graph, model: LabModel[Any], features: str, n_features: int) -> Prediction:
    """O grupo do ponto central mais próximo, se estiver a menos de ``eps``; senão, ruído."""
    memory = _memory(model)
    if not len(memory.rows):
        # Sem pontos centrais, tudo é ruído: -1 em cada linha.
        rows = graph.op("Shape", [features], "forma", end=1)
        noise = graph.op(
            "ConstantOfShape",
            [rows],
            "ruido",
            value=onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64)),
        )
        return Prediction(noise)
    distances = _euclidean(graph, features, memory.rows, "centrais")
    closest = graph.op("ArgMin", [distances], "mais_proximo", axis=1, keepdims=0)
    axes = graph.constant(np.array([1], dtype=np.int64), "eixo_1")
    nearest = graph.op("ReduceMin", [distances, axes], "distancia", keepdims=0)
    eps = graph.constant(np.float64(model.hyperparameters.eps), "eps")
    within = graph.op("LessOrEqual", [nearest, eps], "perto")
    labels = graph.constant(memory.labels.astype(np.int64), "grupos_centrais")
    group = graph.op("Gather", [labels, closest], "grupo_proximo", axis=0)
    noise = graph.constant(np.array(-1, dtype=np.int64), "ruido")
    return Prediction(graph.op("Where", [within, group, noise], "grupo"))


# --- as redes ----------------------------------------------------------------------------


@CONVERTERS.register("mlp")
def _mlp_classifier(
    graph: Graph, model: LabModel[Any], features: str, n_features: int
) -> Prediction:
    """As camadas densas e, no fim, o softmax: ganha a classe com a saída mais alta."""
    logits = _dense(graph, model, features)
    classes = model.classes()
    if classes is None:
        raise CapabilityError("The network didn't keep the training classes.")
    # As probabilidades em double, como no laboratório (o softmax das saídas em float32).
    wide = graph.op("Cast", [logits], "saidas_double", to=DOUBLE)
    probabilities = graph.op("Softmax", [wide], "probabilidades", axis=1)
    index = graph.op("ArgMax", [logits], "indice", axis=1, keepdims=0)
    label = graph.op("Gather", [graph.constant(classes, "classes"), index], "classe", axis=0)
    return Prediction(label, probabilities, tuple(int(value) for value in classes))


@CONVERTERS.register("mlp_regression")
def _mlp_regressor(
    graph: Graph, model: LabModel[Any], features: str, n_features: int
) -> Prediction:
    """As camadas densas, e a saída de volta à escala do alvo (``saída · desvio + média``)."""
    output = graph.op("Cast", [_dense(graph, model, features)], "saida_double", to=DOUBLE)
    state = model.state()
    spread = graph.constant(np.float64(state["target_std"]), "desvio_alvo")
    center = graph.constant(np.float64(state["target_mean"]), "media_alvo")
    scaled = graph.op("Mul", [output, spread], "saida_escalada")
    return Prediction(graph.op("Add", [scaled, center], "previsao"))


def _dense(graph: Graph, model: LabModel[Any], features: str) -> str:
    """A rede, camada a camada (``Gemm`` e a ativação), em ``float32`` como no PyTorch."""
    if not isinstance(model, _Layered):
        raise CapabilityError(f"{model.info.display_name} doesn't expose the network's layers.")
    activations = {"relu": "Relu", "tanh": "Tanh", "sigmoid": "Sigmoid"}
    value = graph.op("Cast", [features], "features_float", to=FLOAT)
    for number, layer in enumerate(model.dense_layers(), start=1):
        weights = graph.constant(layer.weights, f"pesos_{number}")
        bias = graph.constant(layer.bias, f"ordenada_{number}")
        value = graph.op("Gemm", [value, weights, bias], f"camada_{number}", transB=1)
        if layer.activation is not None:
            value = graph.op(activations[layer.activation], [value], f"ativacao_{number}")
    return value


# --- peças comuns ------------------------------------------------------------------------


def _linear_scores(
    graph: Graph, model: LabModel[Any], features: str
) -> tuple[str, NDArray[np.int64]]:
    """``x · pesosᵀ + ordenadas`` de um modelo linear, e as classes dele."""
    state = model.state()
    weights = np.asarray(state["coefficients"], dtype=np.float64)
    bias = np.asarray(state["intercept"], dtype=np.float64)
    product = graph.op("MatMul", [features, graph.constant(weights.T.copy(), "pesos")], "soma")
    scores = graph.op("Add", [product, graph.constant(bias, "ordenadas")], "pontuacoes")
    return scores, np.asarray(state["classes"], dtype=np.int64)


def _euclidean(graph: Graph, features: str, reference: NDArray[np.float64], hint: str) -> str:
    """As distâncias ``(N, M)``, com a fórmula do laboratório: ``√(|a|² - 2 a·b + |b|²)``."""
    axes = graph.constant(np.array([1], dtype=np.int64), "eixo_1")
    squares = graph.op("Mul", [features, features], "quadrados")
    own = graph.op("ReduceSum", [squares, axes], "norma_pontos", keepdims=1)
    cross = graph.op(
        "MatMul", [features, graph.constant(reference.T.copy(), f"{hint}_t")], f"{hint}_produto"
    )
    twice = graph.op("Mul", [graph.constant(np.float64(2.0), "dois"), cross], f"{hint}_dobro")
    norms = graph.constant((reference**2).sum(axis=1), f"{hint}_normas")
    squared = graph.op("Add", [graph.op("Sub", [own, twice], f"{hint}_menos"), norms], f"{hint}_d2")
    zero = graph.constant(np.float64(0.0), "zero")
    positive = graph.op("Max", [squared, zero], f"{hint}_d2_positivo")
    return graph.op("Sqrt", [positive], f"{hint}_distancias")


def _manhattan(graph: Graph, features: str, reference: NDArray[np.float64]) -> str:
    """A soma das diferenças absolutas, ``(N, M)``."""
    axes1 = graph.constant(np.array([1], dtype=np.int64), "eixo_1")
    points = graph.op("Unsqueeze", [features, axes1], "pontos_3d")
    rows = graph.constant(reference[None, :, :].copy(), "treino_3d")
    gaps = graph.op("Abs", [graph.op("Sub", [points, rows], "diferencas")], "distancias_abs")
    axes2 = graph.constant(np.array([2], dtype=np.int64), "eixo_2")
    return graph.op("ReduceSum", [gaps, axes2], "treino_distancias", keepdims=0)


def _vote_weights(graph: Graph, near: str, *, distance: bool) -> str:
    """O peso de cada vizinho: 1, ou ``1 / distância`` (um vizinho à distância zero leva tudo)."""
    zero = graph.constant(np.float64(0.0), "zero")
    one = graph.constant(np.float64(1.0), "um")
    if not distance:
        return graph.op("Add", [graph.op("Mul", [near, zero], "zeros_vizinhos"), one], "pesos")
    exact = graph.op("Equal", [near, zero], "exatos")
    safe = graph.op("Where", [exact, one, near], "distancias_seguras")
    inverse = graph.op("Where", [exact, zero, graph.op("Div", [one, safe], "inversos")], "inv")
    exact_number = graph.op("Cast", [exact], "exatos_num", to=DOUBLE)
    axes = graph.constant(np.array([1], dtype=np.int64), "eixo_1")
    any_exact = graph.op("ReduceMax", [exact_number, axes], "algum_exato", keepdims=1)
    has_exact = graph.op("Greater", [any_exact, zero], "tem_exato")
    return graph.op("Where", [has_exact, exact_number, inverse], "pesos")


def _flat(graph: Graph, value: str) -> str:
    """O tensor ``(N, 1)`` como ``(N,)``."""
    shape = graph.constant(np.array([-1], dtype=np.int64), "plano")
    return graph.op("Reshape", [value, shape], f"{value}_plano")


def _memory(model: LabModel[Any]) -> Memory:
    if not isinstance(model, _Remembering):
        raise CapabilityError(f"{model.info.display_name} doesn't expose the rows it keeps.")
    return model.memory()
