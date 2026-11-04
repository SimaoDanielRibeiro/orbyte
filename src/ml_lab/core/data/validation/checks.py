"""Verificações: ficheiro vazio, colunas repetidas, tamanho e codificação.

A validação responde a uma pergunta: *estes dados são utilizáveis?* Não altera
nada. As verificações do ficheiro (existe, formato, tamanho, codificação) correm
antes de o ler; as da tabela (linhas, colunas, colunas vazias) correm depois.

Os problemas que impedem continuar são erros; os outros são avisos, que a
interface mostra para o utilizador decidir.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from ml_lab.core.foundation.errors import DataSourceError, DataValidationError
from ml_lab.core.foundation.notices import Notice, NoticeLevel, has_errors, sort_notices
from ml_lab.formatting import format_count, format_quantity

type FileFormat = Literal["delimited", "excel", "json", "parquet"]

MAX_FILE_BYTES: Final = 200 * 1024 * 1024
"""Tamanho máximo de um ficheiro (200 MB): o que se lê de uma vez, em memória, sem demorar."""

MIN_ROWS: Final = 10
"""Abaixo deste número de linhas não há dados suficientes para treinar e testar."""

DELIMITED_SUFFIXES: Final = (".csv", ".tsv", ".txt")
EXCEL_SUFFIXES: Final = (".xlsx", ".xlsm", ".xls")
"""O ``.xls`` é o Excel antigo (97 a 2003), que o pandas lê com o ``calamine`` (ADR 0052)."""
JSON_SUFFIXES: Final = (".json", ".jsonl", ".ndjson")
"""Uma lista de registos (``[{"a": 1}, …]``), ou um registo por linha (JSON Lines)."""
PARQUET_SUFFIXES: Final = (".parquet", ".pq")
"""O formato em colunas das ferramentas de dados: já comprimido por dentro (ADR 0052)."""
COMPRESSED_SUFFIX: Final = ".gz"

# Codificações tentadas por ordem. A cp1252 é a do Excel em Windows (em Portugal); a
# latin-1 aceita qualquer sequência de bytes, por isso é o último recurso.
_ENCODINGS: Final = ("utf-8-sig", "cp1252", "latin-1")
_UNNAMED: Final = re.compile(r"Unnamed: \d+")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Resultado da validação: os avisos e se os dados são utilizáveis.

    Attributes:
        notices: Avisos e erros encontrados, do mais grave para o menos grave.
        is_usable: Se não há erros: os dados podem seguir para as etapas seguintes. Calcula-se
            sempre a partir dos avisos (o valor dado ao construtor não conta). É um campo, e não
            uma propriedade, para ir no JSON: é uma regra de domínio, que o frontend lê em vez
            de a decidir.
    """

    notices: tuple[Notice, ...] = ()
    is_usable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_usable", not has_errors(self.notices))

    @property
    def errors(self) -> tuple[Notice, ...]:
        """Só os erros."""
        return tuple(n for n in self.notices if n.level is NoticeLevel.ERROR)

    @property
    def warnings(self) -> tuple[Notice, ...]:
        """Só os avisos (nível ``WARNING``)."""
        return tuple(n for n in self.notices if n.level is NoticeLevel.WARNING)


def file_format(path: Path) -> FileFormat:
    """Formato do ficheiro, pela extensão (``dados.csv.gz`` conta como CSV).

    Raises:
        DataSourceError: Se a extensão não for suportada.
    """
    suffixes = [suffix.lower() for suffix in path.suffixes]
    compressed = bool(suffixes) and suffixes[-1] == COMPRESSED_SUFFIX
    if compressed:
        suffixes.pop()
    last = suffixes[-1] if suffixes else ""
    if last in DELIMITED_SUFFIXES:
        return "delimited"
    if last in JSON_SUFFIXES:
        return "json"
    # O Excel e o Parquet já vêm comprimidos por dentro: um .gz por cima não se lê.
    if last in EXCEL_SUFFIXES and not compressed:
        return "excel"
    if last in PARQUET_SUFFIXES and not compressed:
        return "parquet"
    supported = ", ".join((*DELIMITED_SUFFIXES, *EXCEL_SUFFIXES, *JSON_SUFFIXES, *PARQUET_SUFFIXES))
    raise DataSourceError(
        f"The format of '{path.name}' isn't supported. Use one of these: {supported} (CSV and "
        f"JSON files can also come compressed as {COMPRESSED_SUFFIX})."
    )


def check_file(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> FileFormat:
    """Confirma que o ficheiro existe, tem um formato suportado e um tamanho aceitável.

    Args:
        path: Caminho do ficheiro.
        max_bytes: Tamanho máximo aceite.

    Returns:
        O formato do ficheiro.

    Raises:
        DataSourceError: Se o ficheiro não existir ou o formato não for suportado.
        DataValidationError: Se estiver vazio ou for grande demais.
    """
    if not path.is_file():
        raise DataSourceError(f"The file '{path}' doesn't exist or isn't a file.")
    kind = file_format(path)
    size = path.stat().st_size
    if size == 0:
        raise DataValidationError([f"The file '{path.name}' is empty (0 bytes)."])
    if size > max_bytes:
        raise DataValidationError(
            [
                f"The file '{path.name}' is {_megabytes(size)} MB and the maximum is "
                f"{_megabytes(max_bytes)} MB. Use a sample of the data."
            ]
        )
    return kind


def decode_text(content: bytes) -> tuple[str, tuple[Notice, ...]]:
    """Converte os bytes de um ficheiro de texto em texto, detetando a codificação.

    Tenta UTF-8 (com ou sem BOM), depois Windows-1252 e, por fim, Latin-1. Se não
    for UTF-8, junta um aviso: os acentos podem ter sido mal interpretados.

    Returns:
        O texto e os avisos sobre a codificação.
    """
    for encoding in _ENCODINGS:
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "utf-8-sig":
            return text, ()
        return text, (
            Notice.warning(
                "encoding",
                f"The file isn't in UTF-8: it was read as {encoding}. Check that accented "
                "letters look right; if not, save it again in UTF-8.",
            ),
        )
    raise AssertionError("latin-1 aceita qualquer sequência de bytes")  # pragma: no cover


def check_header(raw_names: Sequence[object]) -> tuple[list[str], tuple[Notice, ...]]:
    """Corrige os nomes do cabeçalho e explica o que mudou.

    - nomes com espaços nas pontas perdem-nos;
    - colunas sem nome passam a ``Unnamed: N`` (como no pandas), com um aviso;
    - nomes repetidos ganham um sufixo (``idade``, ``idade.1``), com um aviso.

    Args:
        raw_names: Os nomes tal como estão na primeira linha do ficheiro.

    Returns:
        Os nomes corrigidos, sem repetições, e os avisos.
    """
    notices: list[Notice] = []
    names: list[str] = []
    for position, raw in enumerate(raw_names):
        missing = raw is None or (isinstance(raw, float) and math.isnan(raw))
        name = "" if missing else str(raw).strip()
        if not name:
            name = f"Unnamed: {position}"
            notices.append(
                Notice.warning(
                    "unnamed_column",
                    f"Column {position + 1} has no name in the header; it became '{name}'. If "
                    "it's an index saved by mistake, drop it in the cleaning.",
                    name,
                )
            )
        names.append(name)

    unique: list[str] = []
    seen: Counter[str] = Counter()
    for name in names:
        candidate = name
        while candidate in seen:
            candidate = f"{name}.{seen[name]}"
            seen[name] += 1
        seen[candidate] += 1
        unique.append(candidate)

    for name, count in Counter(names).items():
        if count > 1:
            copies = [u for u, n in zip(unique, names, strict=True) if n == name][1:]
            notices.append(
                Notice.warning(
                    "duplicate_columns",
                    f"The column '{name}' appears {count} times in the header; the copies became "
                    f"{', '.join(repr(c) for c in copies)}. Check which one is right.",
                    name,
                )
            )
    return unique, tuple(notices)


def validate_table(frame: pd.DataFrame) -> ValidationReport:
    """Verifica se a tabela lida é utilizável: tem linhas, colunas e valores.

    Args:
        frame: A tabela tal como foi lida.

    Returns:
        O relatório com os erros e os avisos.
    """
    n_rows, n_columns = frame.shape
    notices: list[Notice] = []
    if n_columns == 0:
        notices.append(Notice.error("no_columns", "The table has no columns."))
    if n_rows == 0:
        notices.append(
            Notice.error("no_rows", "The table has no data rows, only the header (or nothing).")
        )
    if notices:
        return ValidationReport(tuple(notices))

    if n_rows < MIN_ROWS:
        notices.append(
            Notice.warning(
                "few_rows",
                f"There are only {n_rows} rows: too few to train a model and still have a test "
                "left.",
            )
        )
    if n_columns == 1:
        notices.append(
            Notice.warning(
                "single_column",
                "There's only one column. For supervised learning you need at least one feature "
                "and a target.",
            )
        )
    if n_columns > n_rows:
        notices.append(
            Notice.warning(
                "wide_table",
                f"There are more columns ({n_columns}) than rows ({n_rows}): models easily "
                "memorise data like this (overfitting).",
            )
        )

    missing = frame.isna()
    for name in frame.columns[missing.all(axis=0).to_numpy()]:
        notices.append(
            Notice.warning(
                "empty_column", f"The column '{name}' has no values at all.", subject=str(name)
            )
        )
    empty_rows = int(missing.all(axis=1).sum())
    if empty_rows:
        notices.append(
            Notice.warning(
                "empty_rows",
                f"There are {format_quantity(empty_rows, 'row', 'rows')} with no values at all: "
                "the cleaning can drop them (the drop_missing_rows operation with how='all').",
            )
        )
    for name in frame.columns:
        if _UNNAMED.fullmatch(str(name)) and _looks_like_an_index(frame[name]):
            notices.append(
                Notice.warning(
                    "index_column",
                    f"The column '{name}' is 0, 1, 2…: it looks like the index of a DataFrame "
                    "saved with pandas. Drop it in the cleaning.",
                    str(name),
                )
            )
    return ValidationReport(tuple(notices))


def merge_reports(*parts: Sequence[Notice] | ValidationReport) -> ValidationReport:
    """Junta avisos de várias verificações num só relatório, do mais grave para o menos grave."""
    collected: list[Notice] = []
    for part in parts:
        collected.extend(part.notices if isinstance(part, ValidationReport) else part)
    return ValidationReport(sort_notices(collected))


def _looks_like_an_index(column: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(column):
        return False
    return bool(np.array_equal(column.to_numpy(), np.arange(len(column))))


def _megabytes(size: int) -> str:
    return format_count(round(size / (1024 * 1024)))
