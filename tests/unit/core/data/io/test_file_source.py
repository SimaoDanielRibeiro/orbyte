import gzip
from pathlib import Path

import pandas as pd
import pytest

from ml_lab.core.data.io.base import SourceSpec, open_source
from ml_lab.core.data.io.file_source import FileSource, read_table
from ml_lab.core.foundation.errors import DataSourceError, DataValidationError

FILES = Path(__file__).parent / "files"
"""Os ficheiros que não se conseguem gravar nos testes (o Excel antigo precisava do Excel)."""


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


def test_reads_an_old_excel_file() -> None:
    # Gravado pelo Excel como "Livro do Excel 97-2003" (ADR 0052): o pandas lê-o com o calamine.
    table = read_table(FILES / "livro-antigo.xls")

    assert table.frame.columns.tolist() == ["cidade", "habitantes", "área_km2", "fundação"]
    assert table.frame["cidade"].tolist() == ["Évora", "Porto", "Braga", "Faro"]
    assert table.frame["habitantes"].isna().tolist() == [False, False, True, False]
    assert codes(table.notices) == ["excel_sheets"]


def test_reads_parquet_with_plain_types_and_the_named_index(tmp_path: Path) -> None:
    path = tmp_path / "clientes.parquet"
    pd.DataFrame(
        {
            "plano": pd.Categorical(["Básico", "Família", None]),
            "idade": pd.array([31, None, 45], dtype="Int64"),
            "filhos": pd.array([0, 2, 1], dtype="Int64"),
            "ativo": pd.array([True, None, False], dtype="boolean"),
            "nome": pd.array(["Ana", None, "Rui"], dtype="string"),
            "quando": pd.date_range("2024-01-10 23:00", periods=3, freq="h", tz="Europe/Madrid"),
            "etiquetas": [["a", "b"], [], ["c"]],
        },
        index=pd.Index(["c1", "c2", "c3"], name="cliente"),
    ).to_parquet(path)

    table = read_table(path)

    frame = table.frame
    # O índice com nome volta como coluna; os tipos passam aos de um CSV.
    assert frame.columns.tolist()[0] == "cliente"
    assert frame["plano"].tolist()[:2] == ["Básico", "Família"]
    assert frame["plano"].isna().tolist() == [False, False, True]
    assert frame["idade"].dtype == "float64"
    assert frame["filhos"].dtype == "int64"
    assert frame["ativo"].tolist()[0] is True
    assert frame["ativo"].isna().tolist() == [False, True, False]
    assert frame["nome"].isna().tolist() == [False, True, False]
    assert frame["quando"].dt.tz is None
    assert frame["quando"].dt.hour.tolist() == [23, 0, 1]  # a hora local de cada uma
    assert frame["etiquetas"].tolist() == ['["a", "b"]', "[]", '["c"]']
    assert codes(table.notices) == ["parquet_index"]


def test_a_parquet_index_without_a_name_is_just_the_row_order(tmp_path: Path) -> None:
    path = tmp_path / "filtrado.parquet"
    pd.DataFrame({"a": [1, 2, 3, 4]}).iloc[[1, 3]].to_parquet(path)

    table = read_table(path)

    assert table.frame.to_dict("list") == {"a": [2, 4]}
    assert table.notices == ()


def test_reads_json_records_and_flattens_nested_objects(tmp_path: Path) -> None:
    text = (
        '[{"name": "Ana", "address": {"city": "Évora"}, "tags": ["a", "b"]},'
        ' {"name": "Rui", "address": {"city": "Porto"}, "tags": []}]'
    )

    table = read_table(write(tmp_path / "pessoas.json", text))

    # Os objetos ficam planos; as listas, como texto (uma célula leva um valor).
    assert table.frame.to_dict("list") == {
        "name": ["Ana", "Rui"],
        "address.city": ["Évora", "Porto"],
        "tags": ['["a", "b"]', "[]"],
    }
    assert table.notices == ()


@pytest.mark.parametrize(
    ("filename", "text"),
    [
        ("linhas.jsonl", '{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n'),
        ("api.json", '{"data": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}'),
        ("colunas.json", '{"a": [1, 2], "b": ["x", "y"]}'),
    ],
)
def test_reads_the_other_json_shapes_and_says_how(tmp_path: Path, filename: str, text: str) -> None:
    table = read_table(write(tmp_path / filename, text))

    assert table.frame.to_dict("list") == {"a": [1, 2], "b": ["x", "y"]}
    assert codes(table.notices) == ["json_shape"]


def test_reads_compressed_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "linhas.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'))

    assert read_table(path).frame["a"].tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    ("filename", "content", "error", "message"),
    [
        ("dados.sav", "$FL2", DataSourceError, "isn't supported"),
        ("partido.parquet", "PAR1 isto não é um Parquet", DataValidationError, "as Parquet"),
        ("vazio.json", "{}", DataValidationError, "Couldn't read 'vazio.json' as JSON"),
        ("partido.json", '[{"a": 1}, {"a": 2', DataValidationError, "as JSON"),
        ("numeros.json", "[1, 2, 3]", DataValidationError, "every record must be an object"),
        ("vazio.csv", "", DataValidationError, "is empty"),
        ("brancos.csv", "\n\n  \n", DataValidationError, "has neither a header nor data"),
        ("partido.csv", "a,b\n1,2\n3,4,5,6\n", DataValidationError, "same number of columns"),
    ],
)
def test_unusable_files_are_explained(
    tmp_path: Path, filename: str, content: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        read_table(write(tmp_path / filename, content))


def test_missing_and_too_big_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DataSourceError, match="doesn't exist"):
        read_table(tmp_path / "nada.csv")
    with pytest.raises(DataValidationError, match="the maximum"):
        read_table(write(tmp_path / "grande.csv", "a\n" + "1\n" * 100), max_bytes=10)


def test_file_source_round_trips_through_its_spec(tmp_path: Path) -> None:
    path = write(tmp_path / "dados.csv", "a,b\n1,2\n")
    source = FileSource(str(path), name="Os meus dados")

    table = open_source(source.spec()).load()

    assert source.spec() == SourceSpec("file", {"path": str(path), "name": "Os meus dados"})
    assert table.name == "Os meus dados"
    assert table.frame.shape == (1, 2)
