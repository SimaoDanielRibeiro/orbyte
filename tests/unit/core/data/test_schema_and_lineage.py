import doctest
from types import ModuleType

import pytest

import ml_lab.core.data.lineage as lineage_module
import ml_lab.core.data.schema as schema_module
import ml_lab.core.data.splitting.stratified_split as stratified_module
import ml_lab.core.data.validation.schemas as rules_module
from ml_lab.core.data.lineage import Lineage, LineageStep, Stage
from ml_lab.core.data.schema import ColumnKind, ColumnRole, ColumnSchema, TableSchema
from ml_lab.core.foundation.errors import ColumnNotFoundError, InvalidParameterError


def titanic_schema() -> TableSchema:
    return TableSchema(
        (
            ColumnSchema("PassengerId", ColumnKind.IDENTIFIER, ColumnRole.IGNORED),
            ColumnSchema("Age", ColumnKind.NUMERIC),
            ColumnSchema("Sex", ColumnKind.CATEGORICAL),
            ColumnSchema("Survived", ColumnKind.BOOLEAN),
        )
    )


# --- esquema ----------------------------------------------------------------------------


def test_features_exclude_ignored_columns_and_the_target() -> None:
    schema = titanic_schema().with_target("Survived")

    assert schema.target == ColumnSchema("Survived", ColumnKind.BOOLEAN, ColumnRole.TARGET)
    assert schema.feature_names() == ("Age", "Sex")
    assert schema.feature_names(ColumnKind.NUMERIC) == ("Age",)


def test_changing_the_target_turns_the_old_one_back_into_a_feature() -> None:
    schema = titanic_schema().with_target("Survived").with_target("Age")

    assert schema["Survived"].role is ColumnRole.FEATURE
    assert schema["Age"].role is ColumnRole.TARGET
    assert titanic_schema().with_target("Age").with_target(None).target is None


def test_with_roles_changes_only_the_given_columns() -> None:
    schema = titanic_schema().with_roles({"Sex": ColumnRole.IGNORED})

    assert schema["Sex"].role is ColumnRole.IGNORED
    assert schema["Age"].role is ColumnRole.FEATURE


def test_unknown_columns_are_reported_with_a_suggestion() -> None:
    with pytest.raises(ColumnNotFoundError, match="Querias dizer 'Survived'"):
        titanic_schema().with_target("Survivd")
    with pytest.raises(ColumnNotFoundError):
        titanic_schema().with_roles({"Cabin": ColumnRole.IGNORED})


def test_a_schema_rejects_repeated_names_and_two_targets() -> None:
    with pytest.raises(InvalidParameterError, match="repetidas"):
        TableSchema((ColumnSchema("a", ColumnKind.NUMERIC), ColumnSchema("a", ColumnKind.TEXT)))
    with pytest.raises(InvalidParameterError, match="um alvo"):
        TableSchema(
            (
                ColumnSchema("a", ColumnKind.NUMERIC, ColumnRole.TARGET),
                ColumnSchema("b", ColumnKind.NUMERIC, ColumnRole.TARGET),
            )
        )


def test_free_text_and_identifiers_cannot_feed_a_model() -> None:
    usable = {kind for kind in ColumnKind if kind.is_usable_by_models}

    assert usable == {
        ColumnKind.NUMERIC,
        ColumnKind.CATEGORICAL,
        ColumnKind.BOOLEAN,
        ColumnKind.DATETIME,
    }


# --- linhagem ---------------------------------------------------------------------------


def step(operation: str, stage: Stage = Stage.CLEANING) -> LineageStep:
    return LineageStep(stage, operation, f"{operation} feito", rows_before=10, rows_after=8)


def test_lineage_is_immutable_and_keeps_the_order() -> None:
    first = Lineage().then(step("a"))
    second = first.then(step("b"), step("c", Stage.SPLIT))

    assert [s.operation for s in first] == ["a"]
    assert [s.operation for s in second] == ["a", "b", "c"]
    assert second.for_stage(Stage.SPLIT) == (second[2],)


def test_steps_count_removed_rows_and_become_plain_records() -> None:
    record = step("drop_duplicates").to_record()

    assert step("x").rows_removed == 2
    assert LineageStep(Stage.SPLIT, "split", "dividido").rows_removed == 0
    assert record["stage_label"] == "Limpeza"
    assert record["rows_before"] == 10
    assert record["columns_added"] == []


# --- exemplos das docstrings ----------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [schema_module, lineage_module, rules_module, stratified_module],
    ids=lambda module: module.__name__,
)
def test_docstring_examples_run_as_documented(module: ModuleType) -> None:
    result = doctest.testmod(module)

    assert result.attempted > 0
    assert result.failed == 0
