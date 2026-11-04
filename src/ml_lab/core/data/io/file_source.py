"""Fonte de dados para ficheiros CSV, Excel, JSON e Parquet carregados pelo utilizador.

A leitura faz o trabalho que o utilizador não devia ter de fazer à mão, e explica-o:

- deteta a codificação (UTF-8, Windows-1252 ou Latin-1);
- deteta o separador (``,``, ``;``, tabulação ou ``|``), porque o Excel em
  português grava CSV com ``;``;
- corrige o cabeçalho (nomes em falta ou repetidos) e avisa do que mudou;
- num Excel com várias folhas (também o antigo, ``.xls``), lê a primeira e avisa;
- num JSON, aceita uma lista de registos, JSON Lines e as formas que as APIs gravam, e
  deixa os registos planos (ADR 0051);
- num Parquet, que já traz o tipo de cada coluna, troca os tipos que só o pandas conhece (as
  categorias, os inteiros com nulos, as datas com fuso) pelos de um CSV, que são os que o resto
  do laboratório trata (ADR 0052).

Os valores não são alterados: um número escrito com vírgula decimal (``"1,5"``)
continua texto, e é a limpeza que o converte, com registo na linhagem.
"""

import csv
import gzip
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast, override

import numpy as np
import pandas as pd

from ml_lab.config import get_settings
from ml_lab.core.data.io.base import DATA_SOURCES, DataSource, LoadedTable, SourceSpec
from ml_lab.core.data.validation.checks import (
    COMPRESSED_SUFFIX,
    MAX_FILE_BYTES,
    check_file,
    check_header,
    decode_text,
)
from ml_lab.core.foundation.errors import DataValidationError
from ml_lab.core.foundation.notices import Notice
from ml_lab.formatting import format_list

_DELIMITERS: Final = ",;\t|"
_SNIFF_CHARS: Final = 64 * 1024


@dataclass(frozen=True, slots=True, eq=False)
class TableFile:
    """Uma tabela lida de um ficheiro, com os avisos da leitura."""

    frame: pd.DataFrame
    notices: tuple[Notice, ...]


@DATA_SOURCES.register("file")
@dataclass(frozen=True, slots=True)
class FileSource(DataSource):
    """Um ficheiro CSV, Excel (``.xlsx``, ``.xls``), JSON (``.json``, ``.jsonl``) ou Parquet.

    Attributes:
        path: Caminho do ficheiro; se for relativo, parte da raiz do projeto.
        name: Nome a mostrar; por omissão, o nome do ficheiro.
    """

    path: str
    name: str | None = None

    @override
    def load(self) -> LoadedTable:
        path = Path(self.path)
        if not path.is_absolute():
            path = get_settings().home / path
        table = read_table(path)
        return LoadedTable(
            name=self.name or path.name,
            frame=table.frame,
            spec=self.spec(),
            description=f"Uploaded file: {path.name}.",
            notices=table.notices,
        )

    @override
    def spec(self) -> SourceSpec:
        params: dict[str, str] = {"path": self.path}
        if self.name is not None:
            params["name"] = self.name
        return SourceSpec("file", params)


def read_table(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> TableFile:
    """Lê um ficheiro CSV, Excel, JSON ou Parquet para um DataFrame.

    Args:
        path: Caminho do ficheiro.
        max_bytes: Tamanho máximo aceite (também depois de descomprimir).

    Returns:
        A tabela e os avisos da leitura.

    Raises:
        DataSourceError: Se o ficheiro não existir ou o formato não for suportado.
        DataValidationError: Se o ficheiro estiver vazio, for grande demais ou não se
            conseguir interpretar.
    """
    file_format = check_file(path, max_bytes=max_bytes)
    if file_format == "excel":
        return _read_excel(path)
    if file_format == "json":
        return _read_json(path, max_bytes=max_bytes)
    if file_format == "parquet":
        return _read_parquet(path)
    return _read_delimited(path, max_bytes=max_bytes)


def _read_delimited(path: Path, *, max_bytes: int) -> TableFile:
    content = _read_bytes(path, max_bytes=max_bytes)
    text, notices = decode_text(content)
    if not text.strip():
        raise DataValidationError([f"The file '{path.name}' has neither a header nor data."])
    delimiter = _sniff_delimiter(text, path)
    # Como o pandas, salta as linhas em branco antes do cabeçalho.
    rows = csv.reader(io.StringIO(text), delimiter=delimiter)
    header = next((row for row in rows if any(cell.strip() for cell in row)), [])
    names, header_notices = check_header(header)
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            header=0,
            names=names,
            # Lê o ficheiro inteiro antes de decidir o tipo de cada coluna; em blocos,
            # uma coluna podia ficar com tipos misturados.
            low_memory=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise DataValidationError(
            [
                f"Couldn't read '{path.name}' as CSV ({error}). Check the separator, the quotes "
                "and whether every row has the same number of columns."
            ]
        ) from error
    if delimiter != ",":
        notices = (
            *notices,
            Notice.info("delimiter", f"Separator detected: {_describe_delimiter(delimiter)}."),
        )
    return TableFile(frame, (*notices, *header_notices))


def _read_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.suffix.lower() != COMPRESSED_SUFFIX:
        return path.read_bytes()
    with gzip.open(path, "rb") as compressed:
        # Lê no máximo um byte a mais do que o limite: chega para saber se o ultrapassa.
        content = compressed.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise DataValidationError([f"Once decompressed, '{path.name}' goes over the maximum size."])
    return content


def _sniff_delimiter(text: str, path: Path) -> str:
    sample = text[:_SNIFF_CHARS]
    try:
        return csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        # Acontece, por exemplo, com uma só coluna: não há separador para encontrar.
        return "\t" if ".tsv" in (suffix.lower() for suffix in path.suffixes) else ","


def _describe_delimiter(delimiter: str) -> str:
    names = {";": "semicolon (;)", "\t": "tab", "|": "vertical bar (|)"}
    return names.get(delimiter, repr(delimiter))


def _read_json(path: Path, *, max_bytes: int) -> TableFile:
    """Um JSON com uma lista de registos, ou JSON Lines (um registo por linha).

    Também aceita as formas que as APIs e o pandas gravam: os registos dentro de uma só chave
    (``{"data": [...]}``) e as colunas (``{"idade": [...], "nome": [...]}``). Os objetos dentro
    de um registo ficam planos, com o caminho no nome (``morada.cidade``); as listas ficam como
    texto, porque uma célula só leva um valor.
    """
    content = _read_bytes(path, max_bytes=max_bytes)
    text, notices = decode_text(content)
    if not text.strip():
        raise DataValidationError([f"The file '{path.name}' has no records."])
    try:
        frame, how = _json_frame(text)
    except ValueError as error:
        raise DataValidationError(
            [
                f"Couldn't read '{path.name}' as JSON ({error}). Use a list of records "
                '([{"age": 31, ...}, ...]) or one record per line (JSON Lines).'
            ]
        ) from error
    frame = _nested_as_text(frame)
    names, header_notices = check_header([str(column) for column in frame.columns])
    frame.columns = pd.Index(names)
    if how is not None:
        notices = (*notices, Notice.info("json_shape", how))
    return TableFile(frame, (*notices, *header_notices))


def _json_frame(text: str) -> tuple[pd.DataFrame, str | None]:
    """A tabela de um texto JSON, e como se leu (quando não foi uma lista de registos).

    Raises:
        ValueError: Se o texto não for JSON, ou não tiver a forma de uma tabela.
    """
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as error:
        # Vários objetos seguidos, um por linha: JSON Lines.
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2 or not lines[0].lstrip().startswith("{"):
            raise ValueError(error.msg) from error
        try:
            records = [json.loads(line) for line in lines]
        except json.JSONDecodeError as inner:
            raise ValueError(f"line {inner.lineno}: {inner.msg}") from inner
        return _records(records), "One record per line (JSON Lines)."
    if isinstance(data, list):
        return _records(data), None
    if isinstance(data, dict):
        lists = {key: value for key, value in data.items() if isinstance(value, list)}
        if len(data) == 1 and len(lists) == 1:
            ((key, records),) = lists.items()
            return _records(records), f"The records were read from the key '{key}'."
        if data and all(isinstance(value, list | dict) for value in data.values()):
            return pd.DataFrame(data), "Each key of the object was read as a column."
    raise ValueError("the top level is neither a list of records nor an object of columns")


def _records(records: list[Any]) -> pd.DataFrame:
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("every record must be an object with names and values")
    return pd.json_normalize(records)


def _nested_as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """As células com uma lista ou um objeto (no JSON e no Parquet) passam a texto JSON.

    Uma célula só leva um valor: uma lista de etiquetas fica ``'["a", "b"]'``, como se lê.
    """
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        nested = frame[column].map(lambda value: isinstance(value, list | dict | np.ndarray))
        if nested.any():
            frame[column] = frame[column].map(_as_json_text)
    return frame


def _as_json_text(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _read_parquet(path: Path) -> TableFile:
    """Um ficheiro Parquet: as colunas já trazem o tipo, e não há separador nem codificação.

    O índice que o pandas lá guardou volta como coluna, se tiver nome (o "cliente" de cada
    linha); sem nome, era só a posição das linhas, e sai.
    """
    try:
        frame = pd.read_parquet(path)
    except (ValueError, OSError) as error:
        raise DataValidationError([f"Couldn't read '{path.name}' as Parquet ({error})."]) from error
    notices: list[Notice] = []
    if not isinstance(frame.index, pd.RangeIndex):
        named = [str(name) for name in frame.index.names if name is not None]
        frame = frame.reset_index(drop=not named)
        if named:
            notices.append(
                Notice.info(
                    "parquet_index",
                    f"The index saved in the file ({format_list(named)}) became a column.",
                )
            )
    frame = _nested_as_text(frame.apply(_plain_column))
    names, header_notices = check_header([str(column) for column in frame.columns])
    frame.columns = pd.Index(names)
    return TableFile(frame, (*notices, *header_notices))


def _plain_column(series: pd.Series) -> pd.Series:
    """Os tipos que só o pandas conhece passam aos de um CSV, os que o laboratório trata.

    - as categorias passam aos valores (texto, ou números);
    - os inteiros com nulos (``Int64``) passam a decimais com ``NaN``, como num CSV; sem nulos,
      ficam inteiros;
    - os booleanos com nulos passam a ``True``/``False``/``NaN``;
    - o texto com ``<NA>`` passa ao texto de sempre;
    - as datas com fuso perdem o fuso (ficam com a hora local de cada uma).
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        series = series.astype(dtype.categories.dtype)
        dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        # Os stubs do pandas tipam o acessor `.dt` sem a Series de volta.
        return cast("pd.Series", series.dt.tz_localize(None))
    if not isinstance(dtype, pd.api.extensions.ExtensionDtype) or isinstance(
        dtype, pd.DatetimeTZDtype | pd.PeriodDtype | pd.IntervalDtype
    ):
        return series
    missing = series.isna()
    if pd.api.types.is_bool_dtype(dtype):
        if not missing.any():
            return series.astype(bool)
        return series.astype(object).where(~missing, np.nan)
    if pd.api.types.is_integer_dtype(dtype):
        return series.astype("float64" if missing.any() else "int64")
    if pd.api.types.is_float_dtype(dtype):
        return series.astype("float64")
    if pd.api.types.is_string_dtype(dtype):
        return series.astype(object).where(~missing, np.nan).astype("str")
    return series


def _read_excel(path: Path) -> TableFile:
    notices: list[Notice] = []
    # O motor vai sempre dito: o Excel antigo (.xls) lê-se com o calamine (o xlrd está a sair do
    # pandas), e o .xlsx com o openpyxl de sempre (sem o dizer, o pandas avisa que vai mudar).
    engine: Literal["calamine", "openpyxl"] = (
        "calamine" if path.suffix.lower() == ".xls" else "openpyxl"
    )
    try:
        with pd.ExcelFile(path, engine=engine) as book:
            sheets: Sequence[int | str] = book.sheet_names
            first = sheets[0]
            if len(sheets) > 1:
                notices.append(
                    Notice.info(
                        "excel_sheets",
                        f"The file has {len(sheets)} sheets; the first one ('{first}') was read.",
                    )
                )
            header = pd.read_excel(book, sheet_name=first, header=None, nrows=1)
            raw_names = header.iloc[0].tolist() if len(header) else []
            names, header_notices = check_header(raw_names)
            frame = pd.read_excel(book, sheet_name=first, header=0, names=names)
    except (ValueError, OSError, KeyError) as error:
        raise DataValidationError([f"Couldn't read '{path.name}' as Excel ({error})."]) from error
    return TableFile(frame, (*notices, *header_notices))
