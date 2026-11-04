"""Datasets de exemplo: Iris, Wine, Titanic e California Housing.

Os ficheiros vivem em ``data/samples/`` e estão versionados no Git (ADR 0010); o
script ``scripts/download_samples.py`` volta a descarregá-los das origens. Cada
exemplo declara as regras que os seus dados cumprem, e a leitura confirma-as: um
ficheiro alterado por engano dá um erro claro, em vez de resultados estranhos.

Para acrescentar um exemplo: grava o ficheiro em ``data/samples/``, junta-o ao
script de descarga e regista um ``SampleInfo`` em ``SAMPLES``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from ml_lab.config import get_settings
from ml_lab.core.data.io.base import DATA_SOURCES, DataSource, LoadedTable, SourceSpec
from ml_lab.core.data.io.file_source import read_table
from ml_lab.core.data.validation.schemas import ColumnRule, check_rules
from ml_lab.core.foundation.errors import DataValidationError
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import TaskType
from ml_lab.formatting import format_count


@dataclass(frozen=True, slots=True)
class SampleInfo:
    """Ficha de um dataset de exemplo.

    Attributes:
        name: Identificador, igual ao nome no registo (por exemplo ``"titanic"``).
        title: Nome a mostrar.
        description: O que os dados descrevem e o que têm de interessante para aprender.
        filename: Ficheiro dentro de ``data/samples/``.
        target: Coluna-alvo habitual.
        task_type: Tipo de tarefa habitual com esse alvo.
        source: De onde vêm os dados (autores, publicação).
        url: De onde o script de descarga os obtém.
        n_rows: Número de linhas esperado.
        rules: Regras que as colunas cumprem.
    """

    name: str
    title: str
    description: str
    filename: str
    target: str
    task_type: TaskType
    source: str
    url: str
    n_rows: int
    rules: tuple[ColumnRule, ...]

    @property
    def n_columns(self) -> int:
        """Número de colunas esperado."""
        return len(self.rules)


SAMPLES: Registry[SampleInfo] = Registry("samples", task_types=lambda sample: {sample.task_type})
"""Datasets de exemplo, pelo nome."""


@DATA_SOURCES.register("sample")
@dataclass(frozen=True, slots=True)
class SampleSource(DataSource):
    """Um dataset de exemplo de ``data/samples/``.

    Attributes:
        name: Nome do exemplo no registo ``SAMPLES``.
    """

    name: str

    @override
    def load(self) -> LoadedTable:
        info = SAMPLES.get(self.name)
        table = read_table(sample_path(info))
        problems = [notice.message for notice in check_rules(table.frame, info.rules)]
        if len(table.frame) != info.n_rows:
            problems.append(
                f"Expected {format_count(info.n_rows)} rows and there are "
                f"{format_count(len(table.frame))}."
            )
        if problems:
            raise DataValidationError(
                [
                    f"The sample dataset '{info.name}' doesn't have the expected content.",
                    *problems,
                    "To restore it, run: uv run python scripts/download_samples.py --force",
                ]
            )
        return LoadedTable(
            name=info.title,
            frame=table.frame,
            spec=self.spec(),
            description=info.description,
            suggested_target=info.target,
            notices=table.notices,
        )

    @override
    def spec(self) -> SourceSpec:
        return SourceSpec("sample", {"name": self.name})


def sample_path(info: SampleInfo) -> Path:
    """Caminho do ficheiro de um exemplo."""
    return get_settings().samples_dir / info.filename


def _numeric(*names: str, minimum: float | None = 0.0) -> tuple[ColumnRule, ...]:
    return tuple(
        ColumnRule(name, "numeric", allow_missing=False, minimum=minimum) for name in names
    )


SAMPLES.add(
    "iris",
    SampleInfo(
        name="iris",
        title="Iris",
        description=(
            "Sepal and petal measurements (in cm) of 150 flowers from three iris species, "
            "collected by Edgar Anderson and published by Ronald Fisher in 1936. The target is "
            "species. The data is clean and balanced (50 flowers per species), ideal for a first "
            "classification model."
        ),
        filename="iris.csv",
        target="species",
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        source="R. A. Fisher (1936), copy from the seaborn-data repository",
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        n_rows=150,
        rules=(
            *_numeric("sepal_length", "sepal_width", "petal_length", "petal_width"),
            ColumnRule(
                "species",
                "text",
                allow_missing=False,
                allowed=("setosa", "versicolor", "virginica"),
            ),
        ),
    ),
)

SAMPLES.add(
    "wine",
    SampleInfo(
        name="wine",
        title="Wine",
        description=(
            "Chemical analysis of 178 wines from three grape cultivars of the same region of "
            "Italy: alcohol, acidity, phenols, colour and nine more measurements. The target is "
            "cultivar (1, 2 or 3), an integer with few distinct values, so the suggested task is "
            "classification. The measurements have very different scales (proline reaches the "
            "thousands), which shows why features are scaled."
        ),
        filename="wine.csv",
        target="cultivar",
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        source="M. Forina et al., UCI Machine Learning Repository",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        n_rows=178,
        rules=(
            ColumnRule("cultivar", "integer", allow_missing=False, allowed=("1", "2", "3")),
            *_numeric(
                "alcohol",
                "malic_acid",
                "ash",
                "alcalinity_of_ash",
                "magnesium",
                "total_phenols",
                "flavanoids",
                "nonflavanoid_phenols",
                "proanthocyanins",
                "color_intensity",
                "hue",
                "od280/od315_of_diluted_wines",
                "proline",
            ),
        ),
    ),
)

SAMPLES.add(
    "titanic",
    SampleInfo(
        name="titanic",
        title="Titanic",
        description=(
            "The 891 passengers of Kaggle's training set: class, sex, age, family on board, "
            "fare and port of embarkation. The target is Survived (1 = survived). It has missing "
            "values (Age, Cabin, Embarked), free text (Name, Ticket) and an identifier "
            "(PassengerId): it's the classic example for cleaning and preparing data."
        ),
        filename="titanic.csv",
        target="Survived",
        task_type=TaskType.BINARY_CLASSIFICATION,
        source="Kaggle, “Titanic: Machine Learning from Disaster”, copy from the pandas tutorial",
        url="https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv",
        n_rows=891,
        rules=(
            ColumnRule("PassengerId", "integer", allow_missing=False, minimum=1),
            ColumnRule("Survived", "integer", allow_missing=False, allowed=("0", "1")),
            ColumnRule("Pclass", "integer", allow_missing=False, allowed=("1", "2", "3")),
            ColumnRule("Name", "text", allow_missing=False),
            ColumnRule("Sex", "text", allow_missing=False, allowed=("female", "male")),
            ColumnRule("Age", "numeric", minimum=0, maximum=100),
            ColumnRule("SibSp", "integer", allow_missing=False, minimum=0),
            ColumnRule("Parch", "integer", allow_missing=False, minimum=0),
            ColumnRule("Ticket", "text", allow_missing=False),
            ColumnRule("Fare", "numeric", allow_missing=False, minimum=0),
            ColumnRule("Cabin", "text"),
            ColumnRule("Embarked", "text", allowed=("C", "Q", "S")),
        ),
    ),
)

SAMPLES.add(
    "california_housing",
    SampleInfo(
        name="california_housing",
        title="California Housing",
        description=(
            "The 20,640 districts of California in the 1990 census: location, age of the "
            "houses, rooms, population, median income and closeness to the ocean. The target is "
            "median_house_value (regression). It has 207 missing values in total_bedrooms, a "
            "categorical column and the target capped at 500,001 dollars. Since it's big, the "
            "profile uses a sample."
        ),
        filename="california_housing.csv.gz",
        target="median_house_value",
        task_type=TaskType.REGRESSION,
        source=(
            "R. K. Pace and R. Barry (1997), StatLib; version by A. Géron, “Hands-On Machine "
            "Learning”"
        ),
        url=(
            "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/"
            "housing.csv"
        ),
        n_rows=20_640,
        rules=(
            *_numeric("longitude", "latitude", minimum=None),
            *_numeric("housing_median_age", "total_rooms"),
            ColumnRule("total_bedrooms", "numeric", minimum=0),
            *_numeric("population", "households", "median_income", "median_house_value"),
            ColumnRule(
                "ocean_proximity",
                "text",
                allow_missing=False,
                allowed=("<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"),
            ),
        ),
    ),
)
