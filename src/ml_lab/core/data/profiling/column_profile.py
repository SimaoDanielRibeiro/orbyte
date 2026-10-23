"""Perfil por coluna: nulos, valores únicos e estatísticas.

O perfil só descreve, nunca altera. Para cada coluna diz o tipo inferido (e
porquê), quantos valores faltam, quantos são distintos, as estatísticas e a
distribuição (histograma ou valores mais frequentes), e os avisos que merece.

Os resultados são estruturas simples (números, texto, tuplos), sem DataFrames,
para poderem chegar à interface e à API (ADR 0002).
"""

from dataclasses import dataclass, replace
from typing import Final, cast

import numpy as np
import pandas as pd

from ml_lab.core.data.cleaning.outliers import OutlierBounds, iqr_bounds
from ml_lab.core.data.profiling.type_inference import TypeInference, infer_column
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.foundation.notices import Notice
from ml_lab.formatting import format_count, format_number, format_percent, format_quantity

TOP_VALUES: Final = 10
"""Valores mais frequentes mostrados por coluna."""

MAX_BINS: Final = 50
"""Máximo de barras num histograma."""

MISSING_WARNING: Final = 0.05
"""A partir desta proporção de nulos, a coluna merece um aviso."""

MISSING_SEVERE: Final = 0.5
"""A partir desta proporção, sugere-se remover a coluna."""

SKEW_WARNING: Final = 2.0
"""Assimetria (em valor absoluto) a partir da qual a distribuição é muito enviesada."""

RARE_CATEGORY: Final = 5
"""Categorias com menos linhas do que isto são raras."""


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Estatísticas de uma coluna numérica (calculadas sem os nulos)."""

    mean: float
    std: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    skewness: float | None
    outliers: OutlierBounds | None


@dataclass(frozen=True, slots=True)
class Histogram:
    """Contagens por intervalo: ``counts[i]`` valores entre ``edges[i]`` e ``edges[i + 1]``."""

    edges: tuple[float, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ValueCount:
    """Quantas vezes um valor aparece, e em que proporção das linhas com valor."""

    value: str
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Perfil de uma coluna.

    Attributes:
        name: Nome da coluna.
        kind: Tipo inferido.
        kind_reason: Porque se inferiu esse tipo.
        dtype: Tipo de dados do pandas, por exemplo ``"float64"`` ou ``"str"``.
        n_rows: Número de linhas.
        n_missing: Valores em falta.
        n_unique: Valores distintos (sem contar os nulos).
        stored_as_text: Se são números, datas ou booleanos guardados como texto.
        text_format: Separador decimal ou formato de data, para converter o texto.
        numeric: Estatísticas, para colunas numéricas.
        histogram: Distribuição, para colunas numéricas.
        top_values: Valores mais frequentes, para as restantes colunas.
        date_range: Primeira e última data (ISO 8601), para colunas de datas.
        notices: Avisos sobre esta coluna.
    """

    name: str
    kind: ColumnKind
    kind_reason: str
    dtype: str
    n_rows: int
    n_missing: int
    n_unique: int
    stored_as_text: bool = False
    text_format: str | None = None
    numeric: NumericSummary | None = None
    histogram: Histogram | None = None
    top_values: tuple[ValueCount, ...] = ()
    date_range: tuple[str, str] | None = None
    notices: tuple[Notice, ...] = ()

    @property
    def n_present(self) -> int:
        """Valores que não estão em falta."""
        return self.n_rows - self.n_missing

    @property
    def missing_ratio(self) -> float:
        """Proporção de valores em falta."""
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    @property
    def unique_ratio(self) -> float:
        """Valores distintos a dividir pelos valores presentes."""
        return self.n_unique / self.n_present if self.n_present else 0.0


def profile_column(
    series: pd.Series,
    inference: TypeInference | None = None,
    *,
    full: pd.Series | None = None,
) -> ColumnProfile:
    """Descreve uma coluna.

    Args:
        series: Os valores da coluna (ou de uma amostra das linhas, em datasets grandes).
        inference: O tipo já inferido; se faltar, é inferido aqui.
        full: A coluna completa, quando ``series`` é uma amostra. As contagens (linhas,
            nulos, distintos) vêm daqui, para serem exatas; as estatísticas e a
            distribuição vêm da amostra.
    """
    inference = inference or infer_column(full if full is not None else series)
    name = str(series.name)
    counted = full if full is not None else series
    n_rows = len(counted)
    n_missing = int(counted.isna().sum())
    n_unique = int(counted.nunique())
    present = series.dropna()

    numeric: NumericSummary | None = None
    histogram: Histogram | None = None
    top_values: tuple[ValueCount, ...] = ()
    date_range: tuple[str, str] | None = None
    if inference.kind is ColumnKind.NUMERIC:
        values = _numeric_values(present, inference)
        numeric = _summarize(values)
        histogram = _histogram(values)
    elif inference.kind is ColumnKind.DATETIME:
        dates = _date_values(present, inference)
        if not dates.empty:
            date_range = (dates.min().isoformat(), dates.max().isoformat())
    else:
        top_values = _top_values(present)

    profile = ColumnProfile(
        name=name,
        kind=inference.kind,
        kind_reason=inference.reason,
        dtype=str(series.dtype),
        n_rows=n_rows,
        n_missing=n_missing,
        n_unique=n_unique,
        stored_as_text=inference.stored_as_text,
        text_format=inference.text_format,
        numeric=numeric,
        histogram=histogram,
        top_values=top_values,
        date_range=date_range,
    )
    return _with_notices(profile)


def _numeric_values(present: pd.Series, inference: TypeInference) -> pd.Series:
    if not inference.stored_as_text:
        return present.astype("float64")
    text = present.astype(str).str.strip()
    if inference.text_format == ",":
        text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").dropna().astype("float64")


def _date_values(present: pd.Series, inference: TypeInference) -> pd.Series:
    if not inference.stored_as_text:
        return present
    return pd.to_datetime(
        present.astype(str), format=inference.text_format or "ISO8601", errors="coerce"
    ).dropna()


def _summarize(values: pd.Series) -> NumericSummary | None:
    if values.empty:
        return None
    q1, median, q3 = (float(q) for q in values.quantile([0.25, 0.5, 0.75]))
    # Os stubs do pandas tipam skew() como um escalar qualquer; numa Series float64 é float.
    skewness = float(cast("float", values.skew())) if len(values) > 2 else None
    return NumericSummary(
        mean=float(values.mean()),
        std=float(values.std()) if len(values) > 1 else 0.0,
        minimum=float(values.min()),
        q1=q1,
        median=median,
        q3=q3,
        maximum=float(values.max()),
        skewness=None if skewness is None or np.isnan(skewness) else skewness,
        outliers=iqr_bounds(values),
    )


def _histogram(values: pd.Series) -> Histogram | None:
    if values.empty:
        return None
    array = values.to_numpy(dtype=np.float64)
    edges = np.histogram_bin_edges(array, bins="auto")
    if len(edges) > MAX_BINS + 1:
        edges = np.histogram_bin_edges(array, bins=MAX_BINS)
    counts, edges = np.histogram(array, bins=edges)
    return Histogram(tuple(float(e) for e in edges), tuple(int(c) for c in counts))


def _top_values(present: pd.Series) -> tuple[ValueCount, ...]:
    if present.empty:
        return ()
    counts = present.astype(str).value_counts().head(TOP_VALUES)
    total = len(present)
    return tuple(
        ValueCount(str(value), int(count), int(count) / total) for value, count in counts.items()
    )


def _with_notices(profile: ColumnProfile) -> ColumnProfile:
    notices: list[Notice] = []
    name = profile.name
    if profile.n_present == 0:
        notices.append(
            Notice.warning("empty_column", f"A coluna {name} não tem nenhum valor: remove-a.", name)
        )
        return _replace_notices(profile, notices)

    ratio = profile.missing_ratio
    if profile.n_missing:
        amount = (
            f"{format_percent(ratio)} de valores em falta ({format_count(profile.n_missing)} "
            f"de {format_count(profile.n_rows)})"
        )
        if ratio >= MISSING_SEVERE:
            notices.append(
                Notice.warning(
                    "missing_values",
                    f"A coluna {name} tem {amount}: com tantos nulos, preenchê-los é inventar "
                    "dados; considera removê-la.",
                    name,
                )
            )
        elif ratio >= MISSING_WARNING:
            notices.append(
                Notice.warning(
                    "missing_values",
                    f"A coluna {name} tem {amount}. A transformação pode preenchê-los "
                    "(imputação) com valores aprendidos só no treino.",
                    name,
                )
            )
        else:
            notices.append(Notice.info("missing_values", f"A coluna {name} tem {amount}.", name))

    if profile.n_unique == 1:
        notices.append(
            Notice.warning(
                "constant_column",
                f"A coluna {name} tem sempre o mesmo valor: não ajuda a distinguir nada.",
                name,
            )
        )
    if profile.kind is ColumnKind.IDENTIFIER:
        notices.append(
            Notice.warning(
                "identifier_column",
                f"A coluna {name} parece um identificador (um valor diferente em cada linha): "
                "não ajuda a prever dados novos. Remove-a antes de treinar.",
                name,
            )
        )
    if profile.kind is ColumnKind.TEXT:
        notices.append(
            Notice.info(
                "free_text",
                f"A coluna {name} é texto livre ({format_count(profile.n_unique)} valores "
                "distintos): os modelos não a usam diretamente.",
                name,
            )
        )
    if profile.stored_as_text and profile.kind is not ColumnKind.BOOLEAN:
        notices.append(
            Notice.warning(
                "stored_as_text",
                f"A coluna {name} está guardada como texto, mas os valores são do tipo "
                f"{profile.kind.label.lower()}: converte-a na limpeza.",
                name,
            )
        )
    if profile.numeric is not None:
        notices.extend(_numeric_notices(name, profile.numeric))
    if profile.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
        rare = [v.value for v in profile.top_values if v.count < RARE_CATEGORY]
        if rare:
            notices.append(
                Notice.info(
                    "rare_categories",
                    f"Na coluna {name}, {', '.join(rare)} aparece{'m' if len(rare) > 1 else ''} "
                    f"menos de {RARE_CATEGORY} vezes: pode nem chegar ao treino ou ao teste.",
                    name,
                )
            )
    return _replace_notices(profile, notices)


def _numeric_notices(name: str, summary: NumericSummary) -> list[Notice]:
    notices: list[Notice] = []
    if summary.skewness is not None and abs(summary.skewness) >= SKEW_WARNING:
        side = "direita" if summary.skewness > 0 else "esquerda"
        notices.append(
            Notice.info(
                "skewed",
                f"A distribuição de {name} é muito assimétrica, com cauda para a {side} "
                f"(assimetria {format_number(summary.skewness)}): a média fica longe da mediana.",
                name,
            )
        )
    bounds = summary.outliers
    # Com IQR zero (por exemplo, quase todos os valores iguais a 0), qualquer outro valor
    # seria "extremo": o aviso não ajudaria nada.
    if bounds is not None and bounds.n_outliers and bounds.upper > bounds.lower:
        notices.append(
            Notice.info(
                "outliers",
                f"A coluna {name} tem "
                f"{format_quantity(bounds.n_outliers, 'valor extremo', 'valores extremos')}, fora "
                f"de [{format_number(bounds.lower)}; {format_number(bounds.upper)}] (regra do "
                "IQR). Podem ser reais: confirma antes de os tratar.",
                name,
            )
        )
    return notices


def _replace_notices(profile: ColumnProfile, notices: list[Notice]) -> ColumnProfile:
    return replace(profile, notices=tuple(notices))
