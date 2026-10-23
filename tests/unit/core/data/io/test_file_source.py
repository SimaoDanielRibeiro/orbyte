import gzip
from pathlib import Path

import pandas as pd
import pytest

from ml_lab.core.data.io.base import SourceSpec, open_source
from ml_lab.core.data.io.file_source import FileSource, read_table
from ml_lab.core.foundation.errors import DataSourceError, DataValidationError


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def codes(notices: tuple[object, ...]) -> list[str]:
    return [getattr(notice, "code", "") for notice in notices]


def test_reads_a_comma_separated_file(tmp_path: Path) -> None:
    table = read_table(write(tmp_path / "dados.csv", "idade,nome\n30,Ana\n41,Rui\n"))

    assert table.frame.columns.tolist() == ["idade", "nome"]
    assert table.frame["idade"].tolist() == [30, 41]
    assert table.notices == ()


def test_detects_semicolons_and_keeps_decimal_commas_as_text(tmp_path: Path) -> None:
    # O Excel em português grava assim; a conversão do "1,5" é trabalho da limpeza.
    table = read_table(write(tmp_path / "pt.csv", "preço;nome\n1,5;Ana\n2,25;Rui\n"))

    assert table.frame.columns.tolist() == ["preço", "nome"]
    assert table.frame["preço"].tolist() == ["1,5", "2,25"]
    assert codes(table.notices) == ["delimiter"]


def test_falls_back_to_windows_1252_and_warns(tmp_path: Path) -> None:
    table = read_table(write(tmp_path / "acentos.csv", "cidade\nÉvora\nSão Brás\n", "cp1252"))

    assert table.frame["cidade"].tolist() == ["Évora", "São Brás"]
    assert codes(table.notices) == ["encoding"]


def test_a_utf8_bom_is_not_part_of_the_first_name(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes("idade,nome\n30,Ana\n".encode("utf-8-sig"))

    assert read_table(path).frame.columns.tolist() == ["idade", "nome"]


def test_repeated_and_missing_header_names_are_fixed_and_explained(tmp_path: Path) -> None:
    table = read_table(write(tmp_path / "cabecalho.csv", ",idade,idade\n0,30,31\n1,40,41\n"))

    assert table.frame.columns.tolist() == ["Unnamed: 0", "idade", "idade.1"]
    assert codes(table.notices) == ["unnamed_column", "duplicate_columns"]


def test_reads_compressed_csv(tmp_path: Path) -> None:
    path = tmp_path / "dados.csv.gz"
    path.write_bytes(gzip.compress(b"a,b\n1,2\n3,4\n"))

    assert read_table(path).frame.shape == (2, 2)


def test_reads_the_first_excel_sheet_and_says_so(tmp_path: Path) -> None:
    path = tmp_path / "livro.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_excel(
            writer, sheet_name="Dados", index=False
        )
        pd.DataFrame({"z": [3]}).to_excel(writer, sheet_name="Outra", index=False)

    table = read_table(path)

    assert table.frame.to_dict("list") == {"x": [1, 2], "y": ["a", "b"]}
    assert codes(table.notices) == ["excel_sheets"]


@pytest.mark.parametrize(
    ("filename", "content", "error", "message"),
    [
        ("dados.json", "{}", DataSourceError, "não é suportado"),
        ("vazio.csv", "", DataValidationError, "vazio"),
        ("brancos.csv", "\n\n  \n", DataValidationError, "não tem cabeçalho"),
        ("partido.csv", "a,b\n1,2\n3,4,5,6\n", DataValidationError, "mesmo número de colunas"),
    ],
)
def test_unusable_files_are_explained(
    tmp_path: Path, filename: str, content: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        read_table(write(tmp_path / filename, content))


def test_missing_and_too_big_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DataSourceError, match="não existe"):
        read_table(tmp_path / "nada.csv")
    with pytest.raises(DataValidationError, match="o máximo"):
        read_table(write(tmp_path / "grande.csv", "a\n" + "1\n" * 100), max_bytes=10)


def test_file_source_round_trips_through_its_spec(tmp_path: Path) -> None:
    path = write(tmp_path / "dados.csv", "a,b\n1,2\n")
    source = FileSource(str(path), name="Os meus dados")

    table = open_source(source.spec()).load()

    assert source.spec() == SourceSpec("file", {"path": str(path), "name": "Os meus dados"})
    assert table.name == "Os meus dados"
    assert table.frame.shape == (1, 2)
