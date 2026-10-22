import pytest

from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    ConfigurationError,
    DuplicateComponentError,
    InvalidSeedError,
    MLLabError,
    RegistryError,
)


@pytest.mark.parametrize(
    "error_type",
    [
        ConfigurationError,
        InvalidSeedError,
        RegistryError,
        DuplicateComponentError,
        ComponentNotFoundError,
    ],
)
def test_domain_errors_share_a_common_base(error_type: type[Exception]) -> None:
    assert issubclass(error_type, MLLabError)


def test_domain_errors_are_also_standard_python_errors() -> None:
    assert issubclass(InvalidSeedError, ValueError)
    assert issubclass(ComponentNotFoundError, LookupError)


def test_component_not_found_suggests_the_closest_name() -> None:
    error = ComponentNotFoundError("models", "knm", ["knn", "kmeans"])

    assert "Querias dizer 'knn'?" in str(error)
    assert "Disponíveis: kmeans, knn." in str(error)
    assert error.available == ("kmeans", "knn")


def test_component_not_found_explains_an_empty_registry() -> None:
    error = ComponentNotFoundError("models", "knn", [])

    assert "O registo está vazio" in str(error)


def test_duplicate_component_names_the_conflict() -> None:
    error = DuplicateComponentError("models", "knn")

    assert "'knn'" in str(error)
    assert "'models'" in str(error)
