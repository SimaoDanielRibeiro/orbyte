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
                f"Esperava {format_count(info.n_rows)} linhas e há "
                f"{format_count(len(table.frame))}."
            )
        if problems:
            raise DataValidationError(
                [
                    f"O dataset de exemplo '{info.name}' não tem o conteúdo esperado.",
                    *problems,
                    "Para o repor, corre: uv run python scripts/download_samples.py --force",
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
            "Medidas das sépalas e das pétalas (em cm) de 150 flores de três espécies de íris, "
            "recolhidas por Edgar Anderson e publicadas por Ronald Fisher em 1936. O alvo é "
            "species. São dados limpos e equilibrados (50 flores por espécie), ideais para o "
            "primeiro modelo de classificação."
        ),
        filename="iris.csv",
        target="species",
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        source="R. A. Fisher (1936), cópia do repositório seaborn-data",
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
            "Análise química de 178 vinhos de três castas (cultivares) da mesma região de "
            "Itália: álcool, acidez, fenóis, cor e mais nove medidas. O alvo é cultivar (1, 2 "
            "ou 3), um inteiro com poucos valores distintos, por isso a tarefa sugerida é "
            "classificação. As medidas têm escalas muito diferentes (a prolina chega aos "
            "milhares), o que mostra porque se escalam as features."
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
            "Os 891 passageiros do conjunto de treino do Kaggle: classe, sexo, idade, família a "
            "bordo, tarifa e porto de embarque. O alvo é Survived (1 = sobreviveu). Tem valores "
            "em falta (Age, Cabin, Embarked), texto livre (Name, Ticket) e um identificador "
            "(PassengerId): é o exemplo clássico para limpar e preparar dados."
        ),
        filename="titanic.csv",
        target="Survived",
        task_type=TaskType.BINARY_CLASSIFICATION,
        source="Kaggle, «Titanic: Machine Learning from Disaster», cópia do tutorial do pandas",
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
            "Os 20 640 bairros da Califórnia no censo de 1990: localização, idade das casas, "
            "divisões, população, rendimento mediano e proximidade ao oceano. O alvo é "
            "median_house_value (regressão). Tem 207 valores em falta em total_bedrooms, uma "
            "coluna categórica e o alvo limitado a 500 001 dólares. Por ser grande, o perfil "
            "usa uma amostra."
        ),
        filename="california_housing.csv.gz",
        target="median_house_value",
        task_type=TaskType.REGRESSION,
        source=(
            "R. K. Pace e R. Barry (1997), StatLib; versão de A. Géron, «Hands-On Machine Learning»"
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
