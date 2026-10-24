"""O que o modelo aprendeu, pronto a mostrar (o separador "O que o modelo aprendeu").

Cada modelo descreve o que aprendeu em ``LabModel.state()``: um dicionário com nomes
combinados entre os modelos (``coefficients``, ``intercept``, ``classes``, ``centers``,
``components``, ``tree``, ``feature_importances``…). Este módulo é o único que o lê
para o mostrar. Dá aos pesos o nome da saída que explicam (a previsão, ou uma classe), às
tabelas os nomes das linhas e das colunas, à árvore as suas regras e aos números simples
o seu nome em português. As interfaces (a simples e a aprimorada) só desenham, e a
explicação de uma previsão (``services/analysis.py``) usa os mesmos pesos.

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
        "iterations": "Iterações (ou épocas) do treino",
        "converged": "Convergiu",
        "inertia": "Inércia (soma das distâncias ao quadrado)",
        "n_support": "Vetores de suporte",
        "n_clusters": "Grupos",
        "n_noise": "Pontos de ruído",
        "n_core": "Pontos núcleo",
        "depth": "Profundidade da árvore",
        "n_leaves": "Folhas",
        "n_trees": "Árvores",
        "n_stages": "Etapas",
        "oob_score": "Pontuação fora do saco (OOB)",
        "threshold": "Limiar de anomalia",
        "kl_divergence": "Divergência KL final",
        "log_likelihood": "Log-verosimilhança média",
        "n_terms": "Termos do polinómio",
        "degree": "Grau",
        "n_train": "Exemplos guardados",
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
    """Um número simples do estado (as iterações, a inércia…), com o nome em português."""

    name: str
    label: str
    value: bool | int | float | str


@dataclass(frozen=True, slots=True)
class LearnedView:
    """O que o modelo aprendeu, em partes prontas a mostrar (as que não se aplicam ficam vazias).

    Attributes:
        weights: Num modelo linear, os pesos de cada feature.
        importances: Nas árvores e nos ensembles, a importância de cada feature (pela
            redução da impureza), da maior para a menor.
        tree_rules: Numa árvore, as regras "se … senão …".
        tables: Os centros dos grupos (ou das classes) e as componentes da PCA.
        properties: Os números simples do estado.
        notes: O que vale a pena saber sobre o que não se mostra.
    """

    weights: LinearWeights | None = None
    importances: tuple[tuple[str, float], ...] = ()
    tree_rules: tuple[str, ...] = ()
    tables: tuple[Matrix, ...] = ()
    properties: tuple[Property, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Se não há nada para mostrar além do estado em bruto."""
        parts = (self.importances, self.tree_rules, self.tables, self.properties, self.notes)
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
            f"O modelo tem {count} pesos, um por termo (não por feature): por exemplo, as "
            "potências da regressão polinomial.",
        )
    return LearnedView(
        weights=weights,
        importances=_importances(state, feature_names),
        tree_rules=tree_rules(
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
        meaning = "Um peso positivo empurra para a classe, um negativo afasta."
    else:
        codes = ()
        outputs = (f"Previsão de {target_name}" if target_name else "Previsão",)
        meaning = "Cada peso é quanto a previsão muda quando a feature sobe uma unidade."
    intercepts = np.atleast_1d(np.asarray(state.get("intercept", 0.0), dtype=np.float64))
    return LinearWeights(
        features=tuple(feature_names),
        outputs=outputs,
        codes=codes,
        weights=tuple(tuple(float(weight) for weight in row) for row in matrix),
        intercepts=tuple(float(value) for value in intercepts),
        how_to_read=(
            f"{meaning} Com as features padronizadas, os pesos comparam-se entre si: os "
            "maiores (em valor absoluto) são as features que mais pesam na previsão."
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
            return f"→ {label} ({format_percent(float(values[best]))} de {samples} exemplos)"
        return f"→ {format_number(float(values[0]), 3)} (média de {samples} exemplos)"

    def visit(node: int, depth: int) -> None:
        indent = "    " * depth
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:  # folha
            lines.append(f"{indent}{outcome(node)}")
            return
        if depth >= max_depth:
            lines.append(f"{indent}… (mais {_count_leaves(tree, node)} folhas)")
            return
        name = feature_names[tree["feature"][node]]
        threshold = format_number(float(tree["threshold"][node]), 3)
        lines.append(f"{indent}se {name} ≤ {threshold}:")
        visit(left, depth + 1)
        lines.append(f"{indent}senão ({name} > {threshold}):")
        visit(right, depth + 1)

    visit(0, 0)
    return tuple(lines)


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
            title, rows = "Médias de cada classe", tuple(class_labels[int(c)] for c in codes)
        else:
            title, rows = "Centros de cada grupo", tuple(group_name(g) for g in range(len(centers)))
        tables.append(Matrix(title, rows, columns, centers))
    components = _per_feature(state, "components", feature_names)
    if components:
        rows = tuple(f"Componente {number}" for number in range(1, len(components) + 1))
        tables.append(Matrix("Peso de cada feature em cada componente", rows, columns, components))
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
        if isinstance(value, bool | int | float):
            found.append(Property(name, label, value))
        elif name == "n_support" and isinstance(value, list):  # o SVM, por classe
            counts = [int(count) for count in value]
            written = " + ".join(str(count) for count in counts) + f" = {sum(counts)}"
            found.append(Property(name, label, written))
    return tuple(found)
