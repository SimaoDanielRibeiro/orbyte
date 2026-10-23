"""Fonte de dados para ficheiros CSV e Excel carregados pelo utilizador.

A leitura faz o trabalho que o utilizador não devia ter de fazer à mão, e explica-o:

- deteta a codificação (UTF-8, Windows-1252 ou Latin-1);
- deteta o separador (``,``, ``;``, tabulação ou ``|``), porque o Excel em
  português grava CSV com ``;``;
- corrige o cabeçalho (nomes em falta ou repetidos) e avisa do que mudou;
- num Excel com várias folhas, lê a primeira e avisa.

Os valores não são alterados: um número escrito com vírgula decimal (``"1,5"``)
continua texto, e é a limpeza que o converte, com registo na linhagem.
"""

import csv
import gzip
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

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
    """Um ficheiro CSV (``.csv``, ``.tsv``, ``.txt``, também ``.csv.gz``) ou Excel (``.xlsx``).

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
            description=f"Ficheiro carregado: {path.name}.",
            notices=table.notices,
        )

    @override
    def spec(self) -> SourceSpec:
        params: dict[str, str] = {"path": self.path}
        if self.name is not None:
            params["name"] = self.name
        return SourceSpec("file", params)


def read_table(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> TableFile:
    """Lê um ficheiro CSV ou Excel para um DataFrame.

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
    return _read_delimited(path, max_bytes=max_bytes)


def _read_delimited(path: Path, *, max_bytes: int) -> TableFile:
    content = _read_bytes(path, max_bytes=max_bytes)
    text, notices = decode_text(content)
    if not text.strip():
        raise DataValidationError([f"O ficheiro '{path.name}' não tem cabeçalho nem dados."])
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
                f"Não foi possível ler '{path.name}' como CSV ({error}). Confirma o separador, "
                "as aspas e se todas as linhas têm o mesmo número de colunas."
            ]
        ) from error
    if delimiter != ",":
        notices = (
            *notices,
            Notice.info("delimiter", f"Separador detetado: {_describe_delimiter(delimiter)}."),
        )
    return TableFile(frame, (*notices, *header_notices))


def _read_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.suffix.lower() != COMPRESSED_SUFFIX:
        return path.read_bytes()
    with gzip.open(path, "rb") as compressed:
        # Lê no máximo um byte a mais do que o limite: chega para saber se o ultrapassa.
        content = compressed.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise DataValidationError(
            [f"Depois de descomprimido, '{path.name}' ultrapassa o tamanho máximo."]
        )
    return content


def _sniff_delimiter(text: str, path: Path) -> str:
    sample = text[:_SNIFF_CHARS]
    try:
        return csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        # Acontece, por exemplo, com uma só coluna: não há separador para encontrar.
        return "\t" if ".tsv" in (suffix.lower() for suffix in path.suffixes) else ","


def _describe_delimiter(delimiter: str) -> str:
    names = {";": "ponto e vírgula (;)", "\t": "tabulação", "|": "barra vertical (|)"}
    return names.get(delimiter, repr(delimiter))


def _read_excel(path: Path) -> TableFile:
    notices: list[Notice] = []
    try:
        with pd.ExcelFile(path) as book:
            sheets: Sequence[int | str] = book.sheet_names
            first = sheets[0]
            if len(sheets) > 1:
                notices.append(
                    Notice.info(
                        "excel_sheets",
                        f"O ficheiro tem {len(sheets)} folhas; foi lida a primeira ('{first}').",
                    )
                )
            header = pd.read_excel(book, sheet_name=first, header=None, nrows=1)
            raw_names = header.iloc[0].tolist() if len(header) else []
            names, header_notices = check_header(raw_names)
            frame = pd.read_excel(book, sheet_name=first, header=0, names=names)
    except (ValueError, OSError, KeyError) as error:
        raise DataValidationError(
            [f"Não foi possível ler '{path.name}' como Excel ({error})."]
        ) from error
    return TableFile(frame, (*notices, *header_notices))
