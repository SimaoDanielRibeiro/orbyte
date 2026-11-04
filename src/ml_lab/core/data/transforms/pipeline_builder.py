"""Monta o pipeline de transformações a partir da configuração.

O pipeline aplica as transformações por esta ordem, porque cada uma precisa da
anterior:

1. **Datas** → ano, mês, dia e dia da semana (ou são retiradas, se desligado);
2. **Imputação** dos nulos: numéricas e, depois, categóricas e booleanas;
3. **Intervalos**, nas colunas pedidas (já sem nulos);
4. **Booleanos** → 0/1 e **categóricas** → one-hot;
5. **Polinomiais**, a partir das numéricas;
6. **Escala** das numéricas (as colunas 0/1 do one-hot ficam como estão).

No fim, todas as colunas são números: é o que a etapa de saída converte em NumPy.
"""

from dataclasses import dataclass
from typing import Any, Final, Self

import pandas as pd

from ml_lab.core.data.lineage import LineageStep
from ml_lab.core.data.schema import ColumnKind, TableSchema
from ml_lab.core.data.transforms.base import DropColumns, Transform
from ml_lab.core.data.transforms.encoding import BooleanEncoder, OneHotEncoder
from ml_lab.core.data.transforms.feature_engineering import (
    MAX_DEGREE,
    Binning,
    DateFeatures,
    PolynomialFeatures,
    polynomial_names,
)
from ml_lab.core.data.transforms.imputation import (
    CATEGORICAL_STRATEGIES,
    NUMERIC_STRATEGIES,
    ImputationStrategy,
    Imputer,
)
from ml_lab.core.data.transforms.scaling import MinMaxScaler, ScalingMethod, StandardScaler
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.notices import Notice

MAX_BINS: Final = 20


@dataclass(frozen=True, slots=True)
class TransformConfig:
    """Que transformações aplicar.

    Attributes:
        numeric_imputation: Mediana, média ou ``None`` (não preencher os nulos numéricos).
        categorical_imputation: Valor mais frequente ou ``None``.
        scaling: Padronização, min-max ou ``None`` (não escalar).
        date_features: Se as datas viram ano, mês, dia e dia da semana (senão, saem).
        polynomial_degree: 1 (sem features polinomiais), 2 ou 3.
        binned_columns: Colunas numéricas a dividir em intervalos.
        n_bins: Número de intervalos.
    """

    numeric_imputation: ImputationStrategy | None = ImputationStrategy.MEDIAN
    categorical_imputation: ImputationStrategy | None = ImputationStrategy.MOST_FREQUENT
    scaling: ScalingMethod | None = ScalingMethod.STANDARD
    date_features: bool = True
    polynomial_degree: int = 1
    binned_columns: tuple[str, ...] = ()
    n_bins: int = 5

    def __post_init__(self) -> None:
        # Aceita os valores em texto, como chegam de JSON.
        numeric = _optional(ImputationStrategy, self.numeric_imputation)
        categorical = _optional(ImputationStrategy, self.categorical_imputation)
        object.__setattr__(self, "numeric_imputation", numeric)
        object.__setattr__(self, "categorical_imputation", categorical)
        object.__setattr__(self, "scaling", _optional(ScalingMethod, self.scaling))
        object.__setattr__(self, "binned_columns", tuple(self.binned_columns))
        if numeric is not None and numeric not in NUMERIC_STRATEGIES:
            raise InvalidParameterError("Numeric columns are filled with the mean or the median.")
        if categorical is not None and categorical not in CATEGORICAL_STRATEGIES:
            raise InvalidParameterError(
                "Categorical columns are filled with the most frequent value."
            )
        if not 1 <= self.polynomial_degree <= MAX_DEGREE:
            raise InvalidParameterError(
                f"The polynomial degree must be between 1 and {MAX_DEGREE}."
            )
        if not 2 <= self.n_bins <= MAX_BINS:
            raise InvalidParameterError(f"The number of bins must be between 2 and {MAX_BINS}.")


class TransformPipeline:
    """Uma sequência de transformações aplicadas uma a seguir à outra.

    Tem os mesmos métodos de uma transformação (``fit``, ``transform``,
    ``fit_transform``): aprende com o treino passo a passo, porque cada passo aprende
    com o resultado do anterior.

    Args:
        steps: As transformações, pela ordem em que se aplicam.
        notices: Avisos da montagem (por exemplo, colunas polinomiais a mais).
    """

    def __init__(self, steps: list[Transform], notices: tuple[Notice, ...] = ()) -> None:
        self.steps = steps
        self.notices = notices

    @property
    def is_fitted(self) -> bool:
        """Se todos os passos já aprenderam com o treino."""
        return all(step.is_fitted for step in self.steps)

    def fit(self, frame: pd.DataFrame) -> Self:
        """Aprende cada passo com o resultado do passo anterior, só com ``frame`` (o treino)."""
        current = frame
        for step in self.steps:
            current = step.fit_transform(current)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aplica todos os passos, pela ordem."""
        current = frame
        for step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aprende com ``frame`` e transforma-o."""
        return self.fit(frame).transform(frame)

    def lineage_steps(self) -> tuple[LineageStep, ...]:
        """Um passo de linhagem por transformação (depois do ``fit``)."""
        return tuple(step.lineage_step() for step in self.steps)

    def unknown_categories(self, frame: pd.DataFrame) -> dict[str, int]:
        """Valores de ``frame`` (ainda por transformar) que eram categorias novas no treino."""
        counts: dict[str, int] = {}
        current = frame
        for step in self.steps:
            if isinstance(step, OneHotEncoder):
                counts.update(step.unknown_counts(current))
            current = step.transform(current)
        return counts


def build_pipeline(schema: TableSchema, config: TransformConfig) -> TransformPipeline:
    """Monta o pipeline para as features de ``schema``.

    Args:
        schema: O esquema; só as colunas com o papel de feature entram.
        config: As transformações escolhidas.

    Raises:
        InvalidParameterError: Se houver colunas para dividir em intervalos que não sejam
            features numéricas.
    """
    numeric = list(schema.feature_names(ColumnKind.NUMERIC))
    categorical = list(schema.feature_names(ColumnKind.CATEGORICAL))
    boolean = list(schema.feature_names(ColumnKind.BOOLEAN))
    dates = list(schema.feature_names(ColumnKind.DATETIME))
    binned = list(config.binned_columns)
    not_numeric = [column for column in binned if column not in numeric]
    if not_numeric:
        raise InvalidParameterError(
            f"Only numeric features can be split into bins: {', '.join(not_numeric)} isn't."
        )

    steps: list[Transform] = []
    notices: list[Notice] = []
    date_parts: list[str] = []
    if dates and config.date_features:
        steps.append(DateFeatures(dates))
        date_parts = [name for column in dates for name in DateFeatures.output_names(column)]
    elif dates:
        steps.append(DropColumns(dates, "dates only go into the model as numeric parts."))

    imputable = [*numeric, *date_parts]
    if config.numeric_imputation is not None and imputable:
        steps.append(Imputer(imputable, config.numeric_imputation))
    if config.categorical_imputation is not None and (categorical or boolean):
        steps.append(Imputer([*categorical, *boolean], config.categorical_imputation))
    if binned:
        steps.append(Binning(binned, config.n_bins))
    if boolean:
        steps.append(BooleanEncoder(boolean))
    if categorical:
        steps.append(OneHotEncoder(categorical))

    continuous = [column for column in numeric if column not in binned]
    polynomials = list(polynomial_names(continuous, config.polynomial_degree))
    if polynomials:
        steps.append(PolynomialFeatures(continuous, config.polynomial_degree))
        if len(polynomials) > 100:
            notices.append(
                Notice.warning(
                    "many_polynomial_features",
                    f"The polynomial features create {len(polynomials)} columns: the training "
                    "gets slow and the model memorises easily. Use fewer columns or a lower "
                    "degree.",
                )
            )

    scalable = [*continuous, *date_parts, *polynomials]
    if config.scaling is ScalingMethod.STANDARD and scalable:
        steps.append(StandardScaler(scalable))
    elif config.scaling is ScalingMethod.MINMAX and scalable:
        steps.append(MinMaxScaler(scalable))
    return TransformPipeline(steps, tuple(notices))


def _optional[E: (ImputationStrategy, ScalingMethod)](kind: type[E], value: Any) -> E | None:
    return None if value is None else kind(value)
