"""O que o modelo aprendeu: o estado de cada modelo lido uma só vez, com nomes."""

from typing import Any

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.visualization_data.learned import LearnedView, learned_view

IRIS = SourceSpec("sample", {"name": "iris"})
TITANIC = SourceSpec("sample", {"name": "titanic"})
LINE = SourceSpec("synthetic", {"generator": "regression", "seed": 0, "parameters": {}})


def prepared(source: SourceSpec, **config: Any) -> Dataset:
    return DatasetService().prepared(source, PreparationConfig(**config), 0).dataset


def view(name: str, dataset: Dataset, **values: Any) -> LearnedView:
    model_class = MODELS.get(name)
    model = model_class(model_class.info.hyperparameters.parse(values))
    target = dataset.train.y if dataset.task_type.requires_target else None
    model.fit(dataset.train.X, target)
    return learned_view(
        model.state(),
        task_type=dataset.task_type,
        feature_names=dataset.feature_names,
        class_labels=dataset.class_labels,
        target_name=dataset.target_name,
    )


def test_a_tree_is_written_as_rules_with_its_importances_and_numbers() -> None:
    iris = prepared(IRIS, target="species")

    learned = view("decision_tree", iris, max_depth=2)

    assert learned.tree_rules[0].startswith("if petal")
    assert any(line.strip().startswith("→ setosa") for line in learned.tree_rules)
    importances = [value for _, value in learned.importances]
    assert importances == sorted(importances, reverse=True)
    assert ("Tree depth", 2) in [(p.label, p.value) for p in learned.properties]


def test_a_tree_is_also_given_as_nodes_to_draw() -> None:
    iris = prepared(IRIS, target="species")

    learned = view("decision_tree", iris, max_depth=8)

    nodes = {node.node: node for node in learned.tree}
    root = learned.tree[0]
    assert (root.node, root.depth, root.samples) == (0, 0, len(iris.train))
    assert root.feature is not None
    assert root.feature.startswith("petal")
    for node in learned.tree:
        if node.left is not None and node.right is not None:
            # Cada exemplo do nó vai para um dos dois lados da pergunta.
            assert nodes[node.left].samples + nodes[node.right].samples == node.samples
            assert nodes[node.left].depth == node.depth + 1
        elif node.feature is not None:
            # Uma pergunta abaixo do 4.º nível fica resumida: conta as folhas que esconde.
            assert node.depth == 4
            assert node.hidden_leaves >= 2
    assert max(node.depth for node in learned.tree) <= 4
    leaves = [node for node in learned.tree if node.feature is None]
    assert {node.prediction for node in leaves} <= {"setosa", "versicolor", "virginica"}
    assert all(node.share is not None and 0 < node.share <= 1 for node in learned.tree)


def test_a_regression_tree_predicts_numbers_without_classes() -> None:
    line = prepared(LINE, target="target")

    learned = view("decision_tree_regression", line, max_depth=2)

    assert len(learned.tree) == 7  # 1 + 2 + 4 nós
    assert all(node.share is None and node.code is None for node in learned.tree)
    assert "." in learned.tree[-1].prediction  # um número escrito como no site


def test_the_weights_name_what_each_row_explains() -> None:
    iris = prepared(IRIS, target="species")
    titanic = prepared(TITANIC, target="Survived", cleaning=())
    line = prepared(LINE, target="target")

    multiclass = view("logistic_regression", iris).weights
    binary = view("logistic_regression", titanic).weights
    regression = view("linear_regression", line).weights

    assert multiclass is not None
    assert multiclass.outputs == iris.class_labels
    assert multiclass.row_for(2) == 2
    assert binary is not None
    assert binary.outputs == (titanic.class_labels[1],)  # um só vetor: o da classe positiva
    assert binary.row_for(0) == binary.row_for(1) == 0
    assert regression is not None
    assert regression.outputs == ("Prediction of target",)
    assert regression.codes == ()
    assert len(regression.weights[0]) == line.n_features


def test_polynomial_weights_are_per_term_and_say_so() -> None:
    line = prepared(LINE, target="target")

    learned = view("polynomial_regression", line, degree=3)

    assert learned.weights is None
    assert "one per term" in learned.notes[0]


def test_cluster_centers_are_named_by_group_even_with_a_reference() -> None:
    iris = prepared(IRIS, target="species", task_type=TaskType.CLUSTERING)

    learned = view("kmeans", iris, n_clusters=3)

    centers = learned.tables[0]
    assert centers.rows == ("Group 1", "Group 2", "Group 3")  # não as espécies da referência
    assert centers.columns == iris.feature_names


def test_pca_shows_its_components_and_svm_its_support_vectors() -> None:
    iris = prepared(IRIS, target="species", task_type=TaskType.DIMENSIONALITY_REDUCTION)
    supervised = prepared(IRIS, target="species")

    components = view("pca", iris, n_components=2).tables[0]
    support = next(p for p in view("svm", supervised).properties if p.name == "n_support")

    assert components.rows == ("Component 1", "Component 2")
    assert isinstance(support.value, str)
    assert support.value.count("+") == 2  # um número por classe, e o total
