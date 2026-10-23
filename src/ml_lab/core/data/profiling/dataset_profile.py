"""Perfil global: resumo, correlações e desequilíbrio de classes.

Junta o perfil de cada coluna a uma visão do dataset inteiro: dimensão, memória,
duplicados, nulos (e onde estão), correlações entre colunas numéricas e, quando
há alvo, o equilíbrio das classes e a média de cada feature por classe (por
exemplo, a idade média de quem sobreviveu e de quem não sobreviveu).

Em datasets grandes, as distribuições e as correlações usam uma amostra das linhas
(regra 5 da camada de dados); as contagens de nulos e de duplicados usam todas.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from ml_lab.core.data.profiling.column_profile import ColumnProfile, ValueCount, profile_column
from ml_lab.core.data.profiling.type_inference import (
    TypeInference,
    infer_types,
    looks_like_classes,
)
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.foundation.errors import ColumnNotFoundError
from ml_lab.core.foundation.notices import Notice, sort_notices
from ml_lab.formatting import format_count, format_number, format_percent, format_quantity

DEFAULT_MAX_ROWS: Final = 10_000
"""Acima deste número de linhas, as distribuições usam uma amostra."""

MISSING_MAP_ROWS: Final = 200
"""Linhas mostradas no mapa de nulos."""

HIGH_CORRELATION: Final = 0.9
"""Correlação (em valor absoluto) a partir da qual duas features dizem quase o mesmo."""

LEAKAGE_CORRELATION: Final = 0.95
"""Correlação com o alvo a partir da qual se suspeita de fuga de dados."""

MAX_GROUPS: Final = 20
"""Máximo de classes para calcular as médias por classe."""

FEW_ROWS: Final = 100


@dataclass(frozen=True, slots=True)
class CorrelationMatrix:
    """Correlações de Pearson entre colunas numéricas.

    Attributes:
        columns: As colunas, pela mesma ordem nas linhas e nas colunas da matriz.
        values: ``values[i][j]`` é a correlação entre ``columns[i]`` e ``columns[j]``;
            ``None`` quando não está definida (por exemplo, numa coluna constante).
    """

    columns: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]

    def get(self, first: str, second: str) -> float | None:
        """Correlação entre duas colunas."""
        return self.values[self.columns.index(first)][self.columns.index(second)]


@dataclass(frozen=True, slots=True)
class MissingMap:
    """Onde estão os nulos, numa seleção de linhas espalhadas pelo dataset.

    Attributes:
        columns: As colunas.
        rows: Posição (0, 1, 2…) de cada linha mostrada.
        missing: ``missing[i][j]`` diz se falta o valor da coluna ``j`` na linha ``rows[i]``.
    """

    columns: tuple[str, ...]
    rows: tuple[int, ...]
    missing: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class ClassBalance:
    """Quantas linhas tem cada classe do alvo.

    Attributes:
        target: A coluna-alvo.
        classes: Cada classe e a sua contagem, da mais frequente para a menos frequente.
    """

    target: str
    classes: tuple[ValueCount, ...]

    @property
    def imbalance_ratio(self) -> float:
        """Linhas da classe menos frequente a dividir pelas da mais frequente (1 = equilibrado)."""
        return self.classes[-1].count / self.classes[0].count


@dataclass(frozen=True, slots=True)
class GroupMeans:
    """Média de cada feature numérica, por classe do alvo.

    Attributes:
        target: A coluna-alvo.
        columns: As features.
        groups: As classes.
        means: ``means[i][j]`` é a média de ``columns[j]`` na classe ``groups[i]``.
    """

    target: str
    columns: tuple[str, ...]
    groups: tuple[str, ...]
    means: tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Perfil do dataset inteiro.

    Attributes:
        n_rows: Linhas.
        n_columns: Colunas.
        memory_bytes: Memória ocupada pelo DataFrame.
        n_duplicate_rows: Linhas iguais a uma linha anterior.
        n_missing_cells: Células em falta, em todas as colunas.
        profiled_rows: Linhas usadas nas distribuições (menos do que ``n_rows`` se houve amostra).
        columns: Perfil de cada coluna.
        correlations: Correlações entre colunas numéricas, se houver pelo menos duas.
        missing_map: Onde estão os nulos.
        target: A coluna-alvo usada no perfil, se houver.
        class_balance: Contagem por classe, se o alvo for de classificação.
        group_means: Médias por classe, se o alvo for de classificação.
        notices: Avisos sobre o dataset como um todo.
    """

    n_rows: int
    n_columns: int
    memory_bytes: int
    n_duplicate_rows: int
    n_missing_cells: int
    profiled_rows: int
    columns: tuple[ColumnProfile, ...]
    correlations: CorrelationMatrix | None
    missing_map: MissingMap
    target: str | None = None
    class_balance: ClassBalance | None = None
    group_means: GroupMeans | None = None
    notices: tuple[Notice, ...] = ()

    @property
    def sampled(self) -> bool:
        """Se as distribuições foram calculadas numa amostra."""
        return self.profiled_rows < self.n_rows

    @property
    def missing_ratio(self) -> float:
        """Proporção de células em falta."""
        cells = self.n_rows * self.n_columns
        return self.n_missing_cells / cells if cells else 0.0

    @property
    def kind_counts(self) -> dict[ColumnKind, int]:
        """Quantas colunas há de cada tipo."""
        return dict(Counter(column.kind for column in self.columns))

    @property
    def all_notices(self) -> tuple[Notice, ...]:
        """Os avisos do dataset e de todas as colunas, do mais grave para o menos grave."""
        return sort_notices([*self.notices, *(n for c in self.columns for n in c.notices)])

    def column(self, name: str) -> ColumnProfile:
        """O perfil de uma coluna.

        Raises:
            ColumnNotFoundError: Se a coluna não existir.
        """
        for column in self.columns:
            if column.name == name:
                return column
        raise ColumnNotFoundError(name, [column.name for column in self.columns])


def profile_dataset(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    seed: int = 0,
) -> DatasetProfile:
    """Descreve o dataset.

    Args:
        frame: A tabela; não é alterada.
        target: Coluna-alvo, para o equilíbrio das classes e as médias por classe.
        max_rows: Acima deste número de linhas, as distribuições usam uma amostra.
        seed: Seed da amostra, para o perfil ser sempre o mesmo.

    Raises:
        ColumnNotFoundError: Se ``target`` não existir.
    """
    if target is not None and target not in frame.columns:
        raise ColumnNotFoundError(target, [str(name) for name in frame.columns])
    n_rows = len(frame)
    sample = (
        frame.sample(n=max_rows, random_state=seed).sort_index() if n_rows > max_rows else frame
    )
    inferences = infer_types(frame)
    columns = tuple(
        profile_column(sample[name], inferences[str(name)], full=frame[name])
        for name in frame.columns
    )
    n_duplicates = int(frame.duplicated().sum())

    correlations = _correlations(sample, inferences)
    class_balance = _class_balance(frame, target, inferences) if target else None
    group_means = _group_means(sample, target, inferences) if target and class_balance else None
    notices = _dataset_notices(n_rows, len(sample), n_duplicates)
    if correlations is not None:
        notices.extend(_correlation_notices(correlations, target))
    if class_balance is not None:
        notices.extend(_balance_notices(class_balance))

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=frame.shape[1],
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        n_duplicate_rows=n_duplicates,
        n_missing_cells=int(frame.isna().to_numpy().sum()),
        profiled_rows=len(sample),
        columns=columns,
        correlations=correlations,
        missing_map=_missing_map(frame),
        target=target,
        class_balance=class_balance,
        group_means=group_means,
        notices=tuple(notices),
    )


def _correlations(
    sample: pd.DataFrame, inferences: dict[str, TypeInference]
) -> CorrelationMatrix | None:
    names = [
        name
        for name, inference in inferences.items()
        if inference.kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN)
        and not inference.stored_as_text
        and pd.api.types.is_numeric_dtype(sample[name])
    ]
    if len(names) < 2:
        return None
    matrix = sample[names].astype("float64").corr().to_numpy()
    values = tuple(
        tuple(None if np.isnan(value) else float(value) for value in row) for row in matrix
    )
    return CorrelationMatrix(tuple(names), values)


def _correlation_notices(correlations: CorrelationMatrix, target: str | None) -> list[Notice]:
    notices: list[Notice] = []
    names = correlations.columns
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            value = correlations.get(first, second)
            if value is None:
                continue
            r = format_number(value)
            if target in (first, second):
                feature = second if first == target else first
                if abs(value) >= LEAKAGE_CORRELATION:
                    notices.append(
                        Notice.warning(
                            "possible_leakage",
                            f"{feature} está quase perfeitamente correlacionada com o alvo "
                            f"(r = {r}). Pode ser uma fuga de dados: confirma que a coluna existe "
                            "antes de se saber o resultado.",
                            feature,
                        )
                    )
            elif abs(value) >= HIGH_CORRELATION:
                notices.append(
                    Notice.info(
                        "high_correlation",
                        f"{first} e {second} estão muito correlacionadas (r = {r}): dizem quase "
                        "o mesmo, e alguns modelos ficam instáveis com as duas.",
                        first,
                    )
                )
    return notices


def _class_balance(
    frame: pd.DataFrame, target: str, inferences: dict[str, TypeInference]
) -> ClassBalance | None:
    series = frame[target]
    if not looks_like_classes(series, inferences[target]):
        return None
    present = series.dropna()
    if present.empty:
        return None
    counts = present.astype(str).value_counts()
    total = len(present)
    classes = tuple(
        ValueCount(str(value), int(count), int(count) / total) for value, count in counts.items()
    )
    return ClassBalance(target, classes)


def _balance_notices(balance: ClassBalance) -> list[Notice]:
    notices: list[Notice] = []
    k = len(balance.classes)
    smallest = balance.classes[-1]
    if k >= 2 and smallest.ratio < 0.5 / k:
        notices.append(
            Notice.warning(
                "class_imbalance",
                f"A classe {smallest.value} só tem {format_percent(smallest.ratio)} das linhas "
                f"({format_count(smallest.count)}). Com classes desequilibradas, a exatidão "
                "engana: um modelo que nunca prevê esta classe parece bom. Vê também a recall "
                "e a exatidão equilibrada.",
                balance.target,
            )
        )
    tiny = [c.value for c in balance.classes if c.count < 2]
    if tiny:
        notices.append(
            Notice.warning(
                "tiny_class",
                f"As classes {', '.join(tiny)} têm uma só linha: não podem estar no treino e no "
                "teste ao mesmo tempo, e a divisão estratificada não é possível.",
                balance.target,
            )
        )
    if k == 1:
        notices.append(
            Notice.error(
                "single_class",
                f"O alvo {balance.target} tem um só valor: não há nada para prever.",
                balance.target,
            )
        )
    return notices


def _group_means(
    sample: pd.DataFrame, target: str, inferences: dict[str, TypeInference]
) -> GroupMeans | None:
    columns = [
        name
        for name, inference in inferences.items()
        if name != target
        and inference.kind is ColumnKind.NUMERIC
        and not inference.stored_as_text
        and pd.api.types.is_numeric_dtype(sample[name])
    ]
    if not columns:
        return None
    labels = sample[target].astype(str).where(sample[target].notna())
    grouped = sample[columns].astype("float64").groupby(labels).mean()
    if grouped.empty or len(grouped) > MAX_GROUPS:
        return None
    means = tuple(
        tuple(None if np.isnan(value) else float(value) for value in row)
        for row in grouped.to_numpy()
    )
    return GroupMeans(target, tuple(columns), tuple(str(g) for g in grouped.index), means)


def _missing_map(frame: pd.DataFrame) -> MissingMap:
    n_rows = len(frame)
    if n_rows <= MISSING_MAP_ROWS:
        positions = np.arange(n_rows)
    else:
        positions = np.unique(np.linspace(0, n_rows - 1, MISSING_MAP_ROWS).astype(int))
    missing = frame.iloc[positions].isna().to_numpy()
    return MissingMap(
        columns=tuple(str(name) for name in frame.columns),
        rows=tuple(int(p) for p in positions),
        missing=tuple(tuple(bool(value) for value in row) for row in missing),
    )


def _dataset_notices(n_rows: int, profiled_rows: int, n_duplicates: int) -> list[Notice]:
    notices: list[Notice] = []
    if n_duplicates:
        notices.append(
            Notice.warning(
                "duplicate_rows",
                f"Há {format_quantity(n_duplicates, 'linha duplicada', 'linhas duplicadas')} "
                f"({format_percent(n_duplicates / n_rows)}): a limpeza pode removê-las.",
            )
        )
    if profiled_rows < n_rows:
        notices.append(
            Notice.info(
                "sampled_profile",
                f"As distribuições e as correlações usam uma amostra de "
                f"{format_count(profiled_rows)} das {format_count(n_rows)} linhas; as contagens "
                "de nulos e de duplicados usam todas. O treino usa sempre todas.",
            )
        )
    if 0 < n_rows < FEW_ROWS:
        notices.append(
            Notice.info(
                "few_rows",
                f"Só há {format_count(n_rows)} linhas: os resultados vão variar bastante "
                "consoante a divisão em treino e teste.",
            )
        )
    return notices
