from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.validation.checks import (
    ValidationReport,
    check_header,
    decode_text,
    file_format,
    merge_reports,
    validate_table,
)
from ml_lab.core.data.validation.schemas import ColumnRule, check_rules
from ml_lab.core.foundation.errors import DataSourceError
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.schemas.serialization import from_json, to_json


def codes(notices: tuple[Notice, ...]) -> list[str]:
    return [notice.code for notice in notices]


# --- ficheiro -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dados.csv", "delimited"),
        ("DADOS.TSV", "delimited"),
        ("dados.txt", "delimited"),
        ("dados.csv.gz", "delimited"),
        ("livro.xlsx", "excel"),
        ("livro.xlsm", "excel"),
        ("livro-antigo.XLS", "excel"),
        ("registos.json", "json"),
        ("linhas.jsonl", "json"),
        ("linhas.NDJSON", "json"),
        ("linhas.jsonl.gz", "json"),
        ("colunas.parquet", "parquet"),
        ("colunas.pq", "parquet"),
    ],
)
def test_supported_formats(name: str, expected: str) -> None:
    assert file_format(Path(name)) == expected


# O Excel e o Parquet já vêm comprimidos por dentro: com um .gz por cima, não se leem.
@pytest.mark.parametrize(
    "name",
    ["dados.sav", "livro.xls.gz", "colunas.parquet.gz", "dados.gz", "dados", "livro.xlsx.gz"],
)
def test_unsupported_formats_list_the_alternatives(name: str) -> None:
    with pytest.raises(DataSourceError, match=r"\.csv"):
        file_format(Path(name))


def test_utf8_is_read_without_notices_and_other_encodings_warn() -> None:
    assert decode_text("Évora".encode()) == ("Évora", ())
    text, notices = decode_text("Évora".encode("cp1252"))

    assert text == "Évora"
    assert codes(notices) == ["encoding"]


def test_latin1_accepts_bytes_that_windows_1252_does_not() -> None:
    text, notices = decode_text(b"\x81abc")

    assert text == "\x81abc"
    assert "latin-1" in notices[0].message


def test_header_names_are_stripped_completed_and_made_unique() -> None:
    names, notices = check_header([" idade ", None, "nome", "nome", float("nan"), "nome"])

    assert names == ["idade", "Unnamed: 1", "nome", "nome.1", "Unnamed: 4", "nome.2"]
    assert codes(notices) == ["unnamed_column", "unnamed_column", "duplicate_columns"]
    assert "'nome.1', 'nome.2'" in notices[-1].message


def test_header_suffixes_never_collide_with_existing_names() -> None:
    names, _ = check_header(["a", "a.1", "a"])

    assert len(set(names)) == 3


# --- tabela -----------------------------------------------------------------------------


def test_a_healthy_table_has_no_notices(passengers: pd.DataFrame) -> None:
    report = validate_table(passengers)

    assert report.is_usable
    assert report.notices == ()


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (pd.DataFrame(), ["no_columns", "no_rows"]),
        (pd.DataFrame({"a": []}), ["no_rows"]),
    ],
)
def test_empty_tables_are_errors(frame: pd.DataFrame, expected: list[str]) -> None:
    report = validate_table(frame)

    assert not report.is_usable
    assert codes(report.errors) == expected


def test_whether_the_data_can_go_on_travels_in_the_json() -> None:
    # É uma regra de domínio: o frontend lê-a em vez de a decidir (ADR 0026).
    report = validate_table(pd.DataFrame())

    data = to_json(report)

    assert data["is_usable"] is False
    lying = from_json(ValidationReport, {**data, "is_usable": True})
    assert not lying.is_usable  # calcula-se sempre a partir dos avisos


def test_suspicious_tables_get_warnings() -> None:
    frame = pd.DataFrame(
        {
            "Unnamed: 0": range(5),
            "vazia": [np.nan] * 5,
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [1.0, np.nan, 2.0, 2.0, 3.0],
            "c": [1, 2, 3, 4, 5],
            "d": list("abcde"),
        }
    )
    frame.loc[1, "Unnamed: 0"] = 1  # continua a ser 0, 1, 2…

    report = validate_table(frame)

    assert report.is_usable
    assert set(codes(report.warnings)) == {"few_rows", "wide_table", "empty_column", "index_column"}


def test_rows_without_any_value_are_counted() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan] * 10, "b": [2.0, np.nan] * 10})

    notice = next(n for n in validate_table(frame).notices if n.code == "empty_rows")

    assert "10 rows with no values at all" in notice.message


def test_merged_reports_put_errors_first() -> None:
    report = merge_reports(
        [Notice.info("i", "nota")], (Notice.warning("w", "aviso"), Notice.error("e", "erro"))
    )

    assert [notice.level for notice in report.notices] == [
        NoticeLevel.ERROR,
        NoticeLevel.WARNING,
        NoticeLevel.INFO,
    ]


# --- regras por coluna ---------------------------------------------------------------


def test_rules_report_each_problem(passengers: pd.DataFrame) -> None:
    rules = [
        ColumnRule("Cabin"),
        ColumnRule("Opcional", required=False),
        ColumnRule("Sex", "numeric"),
        ColumnRule("Age", "numeric", allow_missing=False, minimum=5, maximum=50),
        ColumnRule("Embarked", "text", allowed=("S", "C")),
        ColumnRule("Pclass", "integer"),
        ColumnRule("Fare", "integer"),
    ]

    notices = check_rules(passengers, rules)

    assert [(n.code, n.subject) for n in notices] == [
        ("missing_column", "Cabin"),
        ("wrong_type", "Sex"),
        ("unexpected_missing", "Age"),
        ("below_minimum", "Age"),
        ("above_maximum", "Age"),
        ("unexpected_values", "Embarked"),
        ("wrong_type", "Fare"),
    ]


def test_integer_rules_accept_whole_numbers_stored_as_decimals() -> None:
    frame = pd.DataFrame({"n": [1.0, np.nan, 3.0], "b": [True, False, True]})

    assert check_rules(frame, [ColumnRule("n", "integer"), ColumnRule("b", "boolean")]) == ()
