import pytest

from ml_lab.core.foundation.errors import (
    CapabilityError,
    ColumnNotFoundError,
    ComponentNotFoundError,
    ConfigurationError,
    DataError,
    DataSourceError,
    DataValidationError,
    DuplicateComponentError,
    HyperparameterError,
    InvalidParameterError,
    InvalidSeedError,
    MLLabError,
    ModelError,
    NotFittedError,
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
        InvalidParameterError,
        NotFittedError,
        DataError,
        DataSourceError,
        DataValidationError,
        ColumnNotFoundError,
        ModelError,
        HyperparameterError,
        CapabilityError,
    ],
)
def test_domain_errors_share_a_common_base(error_type: type[Exception]) -> None:
    assert issubclass(error_type, MLLabError)


def test_domain_errors_are_also_standard_python_errors() -> None:
    assert issubclass(InvalidSeedError, ValueError)
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(HyperparameterError, ValueError)
    assert issubclass(ComponentNotFoundError, LookupError)
    assert issubclass(ColumnNotFoundError, LookupError)


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


def test_column_not_found_suggests_the_closest_column() -> None:
    error = ColumnNotFoundError("Sruvived", ["Survived", "Age"])

    assert "Querias dizer 'Survived'?" in str(error)
    assert error.column == "Sruvived"
    assert error.available == ("Survived", "Age")


def test_data_validation_error_keeps_every_problem() -> None:
    error = DataValidationError(["O ficheiro está vazio.", "Falta o cabeçalho."])

    assert error.problems == ("O ficheiro está vazio.", "Falta o cabeçalho.")
    assert str(error) == "O ficheiro está vazio. Falta o cabeçalho."


def test_hyperparameter_error_lists_every_problem() -> None:
    error = HyperparameterError("knn", ["k tem de ser pelo menos 1.", "Falta o valor de p."])

    assert error.model == "knn"
    assert error.problems == ("k tem de ser pelo menos 1.", "Falta o valor de p.")
    assert str(error) == (
        "Hiperparâmetros inválidos para 'knn': k tem de ser pelo menos 1. Falta o valor de p."
    )
