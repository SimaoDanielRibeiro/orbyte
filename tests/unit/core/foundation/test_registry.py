import importlib
import sys
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import ClassVar

import pytest

from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    DuplicateComponentError,
    RegistryError,
)
from ml_lab.core.foundation.registry import Registry, import_submodules
from ml_lab.core.foundation.types import TaskType


class Model:
    """Componente de teste: cada subclasse declara os tipos de tarefa que resolve."""

    task_types: ClassVar[frozenset[TaskType]] = frozenset()


def models_registry() -> Registry[type[Model]]:
    return Registry("models", task_types=lambda model: model.task_types)


# --- registar e obter --------------------------------------------------------------


def test_registers_and_gets_a_class_with_the_decorator() -> None:
    registry = models_registry()

    @registry.register("logistic")
    class Logistic(Model):
        task_types = frozenset({TaskType.BINARY_CLASSIFICATION})

    assert registry.get("logistic") is Logistic


def test_decorator_returns_the_component_unchanged() -> None:
    registry: Registry[Callable[[float], float]] = Registry("functions")

    def square(x: float) -> float:
        return x * x

    assert registry.register("square")(square) is square
    assert registry.get("square")(3) == 9


def test_add_registers_objects_without_a_decorator() -> None:
    registry: Registry[object] = Registry("things")
    thing = object()

    assert registry.add("thing", thing) is thing
    assert registry.get("thing") is thing


def test_names_iteration_length_and_membership() -> None:
    registry: Registry[object] = Registry("metrics")
    registry.add("recall", object())
    registry.add("accuracy", object())

    assert registry.names() == ["accuracy", "recall"]
    assert list(registry) == ["accuracy", "recall"]
    assert [name for name, _ in registry.items()] == ["accuracy", "recall"]
    assert len(registry) == 2
    assert "recall" in registry
    assert "f1" not in registry


# --- erros claros --------------------------------------------------------------------


def test_unknown_name_raises_a_clear_error_with_a_suggestion() -> None:
    registry: Registry[object] = Registry("models")
    registry.add("knn", object())
    registry.add("kmeans", object())

    with pytest.raises(ComponentNotFoundError, match="Did you mean 'knn'") as error:
        registry.get("knm")

    assert error.value.registry == "models"
    assert error.value.available == ("kmeans", "knn")


def test_duplicate_names_are_rejected() -> None:
    registry: Registry[object] = Registry("models")
    registry.add("knn", object())

    with pytest.raises(DuplicateComponentError, match="'knn'"):
        registry.add("knn", object())


@pytest.mark.parametrize("name", ["", "KNN", "extra-trees", "1knn", "knn ", "árvore"])
def test_invalid_names_are_rejected(name: str) -> None:
    registry: Registry[object] = Registry("models")

    with pytest.raises(RegistryError, match="Invalid name"):
        registry.add(name, object())


# --- filtrar por tipo de tarefa -----------------------------------------------------


def test_filters_components_by_task_type() -> None:
    registry = models_registry()

    @registry.register("logistic")
    class Logistic(Model):
        task_types = frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION})

    @registry.register("linear")
    class Linear(Model):
        task_types = frozenset({TaskType.REGRESSION})

    @registry.register("decision_tree")
    class DecisionTree(Model):
        task_types = frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION})

    assert registry.for_task(TaskType.REGRESSION) == {
        "decision_tree": DecisionTree,
        "linear": Linear,
    }
    assert registry.for_task(TaskType.BINARY_CLASSIFICATION) == {
        "decision_tree": DecisionTree,
        "logistic": Logistic,
    }
    assert registry.for_task(TaskType.CLUSTERING) == {}
    assert registry.task_types_of("linear") == {TaskType.REGRESSION}


def test_filtering_needs_a_task_types_reader() -> None:
    registry: Registry[object] = Registry("transforms")

    with pytest.raises(RegistryError, match="isn't organized by task type"):
        registry.for_task(TaskType.REGRESSION)


def test_component_without_task_types_is_rejected() -> None:
    registry = models_registry()

    with pytest.raises(RegistryError, match="declares no task type"):
        registry.add("empty", Model)


def _read_missing_card(component: object) -> Iterable[TaskType]:
    raise AttributeError("o componente não tem ficha técnica")


def test_errors_reading_task_types_are_explained() -> None:
    registry: Registry[object] = Registry("models", task_types=_read_missing_card)

    with pytest.raises(RegistryError, match="Couldn't read the task types of 'knn'"):
        registry.add("knn", object())


# --- descoberta automática ------------------------------------------------------------


@pytest.fixture
def plugin_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Pacote temporário cujos módulos se registam num registo com descoberta."""
    name = f"fake_plugins_{uuid.uuid4().hex[:8]}"
    files = {
        "__init__.py": "",
        "catalog.py": (
            "from ml_lab.core.foundation.registry import Registry\n"
            f"PLUGINS = Registry('plugins', discover={name!r})\n"
        ),
        "alpha.py": (
            f"from {name}.catalog import PLUGINS\n\n"
            "@PLUGINS.register('alpha')\n"
            "class Alpha:\n"
            "    pass\n"
        ),
        "nested/__init__.py": "",
        "nested/beta.py": (
            f"from {name}.catalog import PLUGINS\n\n"
            "@PLUGINS.register('beta')\n"
            "class Beta:\n"
            "    pass\n"
        ),
    }
    for relative, content in files.items():
        path = tmp_path / name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.syspath_prepend(tmp_path)
    yield name
    for module in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
        del sys.modules[module]


def add_plugin_module(package: str, filename: str, content: str) -> None:
    root = Path(importlib.import_module(package).__file__ or "").parent
    (root / filename).write_text(content, encoding="utf-8")
    importlib.invalidate_caches()


def test_discovery_imports_every_module_of_the_package_on_first_use(plugin_package: str) -> None:
    plugins: Registry[type] = importlib.import_module(f"{plugin_package}.catalog").PLUGINS

    # Criar o registo não importa nada; a descoberta só corre na primeira consulta.
    assert f"{plugin_package}.alpha" not in sys.modules
    assert plugins.names() == ["alpha", "beta"]
    assert f"{plugin_package}.nested.beta" in sys.modules


def test_import_submodules_lists_the_imported_modules_in_order(plugin_package: str) -> None:
    assert import_submodules(plugin_package) == [
        f"{plugin_package}.alpha",
        f"{plugin_package}.catalog",
        f"{plugin_package}.nested",
        f"{plugin_package}.nested.beta",
    ]


def test_discovery_does_not_hide_import_errors(plugin_package: str) -> None:
    add_plugin_module(plugin_package, "broken.py", "raise RuntimeError('módulo partido')\n")
    plugins: Registry[type] = importlib.import_module(f"{plugin_package}.catalog").PLUGINS

    with pytest.raises(RuntimeError, match="módulo partido"):
        plugins.names()


def test_a_module_can_query_the_registry_while_it_is_being_discovered(
    plugin_package: str,
) -> None:
    # Sem o lock reentrante, esta consulta a meio da descoberta bloqueava o teste.
    add_plugin_module(
        plugin_package,
        "gamma.py",
        f"from {plugin_package}.catalog import PLUGINS\n\nSEEN_DURING_IMPORT = PLUGINS.names()\n",
    )
    plugins: Registry[type] = importlib.import_module(f"{plugin_package}.catalog").PLUGINS

    assert plugins.names() == ["alpha", "beta"]
    gamma = importlib.import_module(f"{plugin_package}.gamma")
    assert gamma.SEEN_DURING_IMPORT == ["alpha"]
