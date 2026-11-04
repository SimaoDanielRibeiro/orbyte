"""O que o modelo aprendeu, pronto a mostrar (a secção "O que aprendeu").

Cada modelo descreve o que aprendeu em ``LabModel.state()``: um dicionário com nomes
combinados entre os modelos (``coefficients``, ``intercept``, ``classes``, ``centers``,
``components``, ``tree``, ``feature_importances``…). Este módulo é o único que o lê
para o mostrar. Dá aos pesos o nome da saída que explicam (a previsão, ou uma classe), às
tabelas os nomes das linhas e das colunas, à árvore as suas regras e aos números simples
o seu nome em inglês. A interface só desenha, e a explicação de uma previsão
(``services/analysis.py``) usa os mesmos pesos.

Os valores estão na escala depois da preparação (padronizados, se a preparação os
padronizou).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from ml_lab.core.evaluation.clustering import group_name
from ml_lab.core.foundation.types import TaskType
from ml_lab.formatting import format_number, format_percent

TREE_RULES_DEPTH: Final = 4
"""Níveis da árvore escritos como regras (os mais fundos ficam resumidos)."""

PROPERTY_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "iterations": "Training iterations (or epochs)",
        "converged": "Converged",
        "inertia": "Inertia (sum of squared distances)",
        "n_support": "Support vectors",
        "n_clusters": "Clusters",
        "n_noise": "Noise points",
        "n_core": "Core points",
        "depth": "Tree depth",
        "n_leaves": "Leaves",
        "n_trees": "Trees",
        "n_stages": "Stages",
        "oob_score": "Out-of-bag score (OOB)",
        "threshold": "Anomaly threshold",
        "kl_divergence": "Final KL divergence",
        "log_likelihood": "Mean log-likelihood",
        "n_terms": "Polynomial terms",
        "degree": "Degree",
        "n_train": "Stored examples",
        "architecture": "Architecture (neurons per layer)",
        "activation": "Hidden-layer activation",
        "n_parameters": "Parameters (weights and intercepts)",
        "best_epoch": "Best epoch (on the validation)",
        "early_stopped": "Stopped early (early stopping)",
        "target_mean": "Target mean (to standardise it)",
        "target_std": "Target standard deviation",
    }
)
"""O nome a mostrar dos números simples do estado (os outros aparecem pelo nome técnico)."""


@dataclass(frozen=True, slots=True)
class LinearWeights:
    """Os pesos de um modelo linear.

    A previsão (ou a pontuação de uma classe) é a ordenada mais a soma de peso × valor de
    cada feature: cada parcela é o contributo exato dessa feature.

    Attributes:
        features: As features, pela ordem dos pesos.
        outputs: O que explica cada linha de pesos: a previsão do alvo, ou uma classe.
        codes: Na classificação, o número da classe de cada linha (vazio na regressão).
        weights: ``weights[i][j]`` é o peso da feature ``j`` na saída ``i``.
        intercepts: A ordenada de cada saída.
        how_to_read: Como se leem estes pesos.
    """

    features: tuple[str, ...]
    outputs: tuple[str, ...]
    codes: tuple[int, ...]
    weights: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    how_to_read: str

    def row_for(self, code: int) -> int | None:
        """A linha que explica a classe ``code`` (na binária há uma só: a da classe positiva)."""
        if len(self.weights) == 1:
            return 0
        return self.codes.index(code) if code in self.codes else None

    def intercept_for(self, row: int) -> float | None:
        """A ordenada da linha ``row``, se o modelo a tiver."""
        return self.intercepts[row] if row < len(self.intercepts) else None


@dataclass(frozen=True, slots=True)
class Matrix:
    """Uma tabela de números com nome nas linhas e nas colunas (os centros, as componentes)."""

    title: str
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class Property:
    """Um número simples do estado (as iterações, a inércia…), com o nome em inglês."""

    name: str
    label: str
    value: bool | int | float | str


@dataclass(frozen=True, slots=True)
class TreeNode:
    """Um nó da árvore de decisão, para a desenhar.

    Um nó de decisão faz uma pergunta ("``feature`` ≤ ``threshold``?"): o "sim" vai para
    ``left`` e o "não" para ``right``. Uma folha dá a previsão.

    Attributes:
        node: O número do nó (a raiz é o 0).
        depth: A profundidade (a raiz tem 0).
        samples: Quantos exemplos de treino chegam a este nó.
        prediction: O que o nó prevê, já escrito: a classe mais comum, ou a média do alvo.
        share: Na classificação, a parte dos exemplos do nó que são dessa classe (de 0 a 1).
        code: Na classificação, o número dessa classe (para a cor).
        feature: Num nó de decisão, a feature da pergunta.
        threshold: Num nó de decisão, o limiar da pergunta.
        left: O nó do "sim", se for mostrado.
        right: O nó do "não", se for mostrado.
        hidden_leaves: Quantas folhas ficam por baixo deste nó sem ser mostradas (a árvore
            desenha-se até ``TREE_RULES_DEPTH`` níveis).
    """

    node: int
    depth: int
    samples: int
    prediction: str
    share: float | None = None
    code: int | None = None
    feature: str | None = None
    threshold: float | None = None
    left: int | None = None
    right: int | None = None
    hidden_leaves: int = 0


@dataclass(frozen=True, slots=True)
class LearnedView:
    """O que o modelo aprendeu, em partes prontas a mostrar (as que não se aplicam ficam vazias).

    Attributes:
        weights: Num modelo linear, os pesos de cada feature.
        importances: Nas árvores e nos ensembles, a importância de cada feature (pela
            redução da impureza), da maior para a menor.
        tree_rules: Numa árvore, as regras "se … senão …".
        tree: Numa árvore, os nós até ``TREE_RULES_DEPTH`` níveis, para a desenhar (a raiz
            primeiro).
        tables: Os centros dos grupos (ou das classes) e as componentes da PCA.
        properties: Os números simples do estado.
        notes: O que vale a pena saber sobre o que não se mostra.
    """

    weights: LinearWeights | None = None
    importances: tuple[tuple[str, float], ...] = ()
    tree_rules: tuple[str, ...] = ()
    tree: tuple[TreeNode, ...] = ()
    tables: tuple[Matrix, ...] = ()
    properties: tuple[Property, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Se não há nada para mostrar além do estado em bruto."""
        parts = (
            self.importances,
            self.tree_rules,
            self.tree,
            self.tables,
            self.properties,
            self.notes,
        )
        return self.weights is None and not any(parts)


def learned_view(
    state: Mapping[str, Any],
    *,
    task_type: TaskType,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    target_name: str | None,
) -> LearnedView:
    """O que o modelo aprendeu, a partir do seu estado e dos nomes dos dados."""
    weights = linear_weights(
        state,
        task_type=task_type,
        feature_names=feature_names,
        class_labels=class_labels,
        target_name=target_name,
    )
    notes: tuple[str, ...] = ()
    coefficients = state.get("coefficients")
    if coefficients and weights is None:
        count = np.atleast_2d(np.asarray(coefficients)).shape[1]
        notes = (
            f"The model has {count} weights, one per term (not per feature): for example, the "
            "powers of polynomial regression.",
        )
    return LearnedView(
        weights=weights,
        importances=_importances(state, feature_names),
        tree_rules=tree_rules(
            state, task_type=task_type, feature_names=feature_names, class_labels=class_labels
        ),
        tree=tree_nodes(
            state, task_type=task_type, feature_names=feature_names, class_labels=class_labels
        ),
        tables=_tables(state, task_type, feature_names, class_labels),
        properties=_properties(state),
        notes=notes,
    )


def linear_weights(
    state: Mapping[str, Any],
    *,
    task_type: TaskType,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    target_name: str | None,
) -> LinearWeights | None:
    """Os pesos de um modelo linear, se o estado tiver um peso por feature."""
    coefficients = state.get("coefficients")
    if not coefficients:
        return None
    matrix = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    if matrix.shape[1] != len(feature_names):
        return None  # um peso por termo, e não por feature (a regressão polinomial)
    if task_type.is_classification:
        codes = tuple(int(code) for code in state.get("classes", range(len(class_labels))))
        if len(matrix) == 1:  # binária: um só vetor, o da classe positiva (a segunda)
            codes = codes[-1:]
        if len(codes) != len(matrix):
            return None
        outputs = tuple(class_labels[code] for code in codes)
        meaning = "A positive weight pushes towards the class, a negative one away from it."
    else:
        codes = ()
        outputs = (f"Prediction of {target_name}" if target_name else "Prediction",)
        meaning = "Each weight is how much the prediction changes when the feature goes up by one."
    intercepts = np.atleast_1d(np.asarray(state.get("intercept", 0.0), dtype=np.float64))
    return LinearWeights(
        features=tuple(feature_names),
        outputs=outputs,
        codes=codes,
        weights=tuple(tuple(float(weight) for weight in row) for row in matrix),
        intercepts=tuple(float(value) for value in intercepts),
        how_to_read=(
            f"{meaning} With standardised features, the weights can be compared with each "
            "other: the largest (in absolute value) are the features that weigh most in the "
            "prediction."
        ),
    )


def tree_rules(
    state: Mapping[str, Any],
    *,
    task_type: TaskType,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    max_depth: int = TREE_RULES_DEPTH,
) -> tuple[str, ...]:
    """A árvore de decisão escrita como regras "se … senão …", até ``max_depth`` níveis.

    Devolve um tuplo vazio se o modelo não for uma árvore.
    """
    tree = state.get("tree")
    if not tree:
        return ()
    lines: list[str] = []

    def outcome(node: int) -> str:
        values = tree["value"][node]
        samples = tree["samples"][node]
        if task_type.is_classification:
            classes = state.get("classes", list(range(len(values))))
            best = int(np.argmax(values))
            label = class_labels[int(classes[best])]
            return f"→ {label} ({format_percent(float(values[best]))} of {samples} examples)"
        return f"→ {format_number(float(values[0]), 3)} (mean of {samples} examples)"

    def visit(node: int, depth: int) -> None:
        indent = "    " * depth
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:  # folha
            lines.append(f"{indent}{outcome(node)}")
            return
        if depth >= max_depth:
            lines.append(f"{indent}… ({_count_leaves(tree, node)} more leaves)")
            return
        name = feature_names[tree["feature"][node]]
        threshold = format_number(float(tree["threshold"][node]), 3)
        lines.append(f"{indent}if {name} ≤ {threshold}:")
        visit(left, depth + 1)
        lines.append(f"{indent}else ({name} > {threshold}):")
        visit(right, depth + 1)

    visit(0, 0)
    return tuple(lines)


def tree_nodes(
    state: Mapping[str, Any],
    *,
    task_type: TaskType,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    max_depth: int = TREE_RULES_DEPTH,
) -> tuple[TreeNode, ...]:
    """Os nós da árvore até ``max_depth`` níveis, para a desenhar (vazio se não for uma árvore).

    Os mesmos números das regras de ``tree_rules``: o que o desenho mostra e o que as regras
    dizem é a mesma árvore.
    """
    tree = state.get("tree")
    if not tree:
        return ()
    nodes: list[TreeNode] = []

    def visit(node: int, depth: int) -> None:
        values = tree["value"][node]
        share: float | None = None
        code: int | None = None
        if task_type.is_classification:
            classes = state.get("classes", list(range(len(values))))
            best = int(np.argmax(values))
            code = int(classes[best])
            prediction, share = class_labels[code], float(values[best])
        else:
            prediction = format_number(float(values[0]), 3)
        samples = int(tree["samples"][node])
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:  # folha
            nodes.append(TreeNode(node, depth, samples, prediction, share, code))
            return
        shown = depth < max_depth
        nodes.append(
            TreeNode(
                node,
                depth,
                samples,
                prediction,
                share,
                code,
                feature=feature_names[tree["feature"][node]],
                threshold=float(tree["threshold"][node]),
                left=left if shown else None,
                right=right if shown else None,
                hidden_leaves=0 if shown else _count_leaves(tree, node),
            )
        )
        if shown:
            visit(left, depth + 1)
            visit(right, depth + 1)

    visit(0, 0)
    return tuple(nodes)


def _count_leaves(tree: Mapping[str, Any], node: int) -> int:
    left, right = tree["left"][node], tree["right"][node]
    if left == -1:
        return 1
    return _count_leaves(tree, left) + _count_leaves(tree, right)


def _importances(
    state: Mapping[str, Any], feature_names: Sequence[str]
) -> tuple[tuple[str, float], ...]:
    values = state.get("feature_importances")
    if not values or len(values) != len(feature_names):
        return ()
    pairs = ((name, float(value)) for name, value in zip(feature_names, values, strict=True))
    return tuple(sorted(pairs, key=lambda pair: -pair[1]))


def _tables(
    state: Mapping[str, Any],
    task_type: TaskType,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
) -> tuple[Matrix, ...]:
    tables: list[Matrix] = []
    columns = tuple(feature_names)
    centers = _per_feature(state, "centers", feature_names)
    if centers:
        if task_type.is_classification:
            codes = state.get("classes", range(len(centers)))
            title, rows = "Means of each class", tuple(class_labels[int(c)] for c in codes)
        else:
            title, rows = (
                "Centres of each cluster",
                tuple(group_name(g) for g in range(len(centers))),
            )
        tables.append(Matrix(title, rows, columns, centers))
    components = _per_feature(state, "components", feature_names)
    if components:
        rows = tuple(f"Component {number}" for number in range(1, len(components) + 1))
        tables.append(Matrix("Weight of each feature in each component", rows, columns, components))
    return tuple(tables)


def _per_feature(
    state: Mapping[str, Any], key: str, feature_names: Sequence[str]
) -> tuple[tuple[float, ...], ...]:
    """As linhas de ``state[key]``, se tiverem uma coluna por feature (senão, nenhuma)."""
    rows = state.get(key) or []
    if not rows or len(rows[0]) != len(feature_names):
        return ()
    return tuple(tuple(float(value) for value in row) for row in rows)


def _properties(state: Mapping[str, Any]) -> tuple[Property, ...]:
    found = []
    for name, value in state.items():
        label = PROPERTY_LABELS.get(name, name)
        if isinstance(value, bool | int | float | str):
            found.append(Property(name, label, value))
        elif name == "n_support" and isinstance(value, list):  # o SVM, por classe
            counts = [int(count) for count in value]
            written = " + ".join(str(count) for count in counts) + f" = {sum(counts)}"
            found.append(Property(name, label, written))
    return tuple(found)
