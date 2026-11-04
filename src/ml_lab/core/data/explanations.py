"""Explicações didáticas de cada etapa da camada de dados.

São os textos do painel "Porquê?" da interface. Vivem no backend, escritos uma só vez (em
inglês, como todo o site; ADR 0043), e a interface pede-os pela API.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from ml_lab.core.foundation.errors import ComponentNotFoundError


@dataclass(frozen=True, slots=True)
class Explanation:
    """Um tema explicado.

    Attributes:
        title: O título.
        text: A explicação, em Markdown.
    """

    title: str
    text: str


EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "input": Explanation(
            "Input",
            "The data arrives as a table: each row is an example (a passenger, a flower) and "
            "each column a characteristic. It can come from a file of yours, a sample dataset or "
            "a synthetic data generator, where you control the shape of the data and the noise. "
            "The original table is never changed: each step works on a copy.",
        ),
        "validation": Explanation(
            "Validation",
            "First of all, the data is checked for whether it can be used: whether the file has "
            "rows and columns, whether the header has repeated names, whether the text encoding "
            "was detected correctly. Validation doesn't change anything; it only warns.",
        ),
        "profiling": Explanation(
            "Profile",
            "The profile describes each column: its type (what the column *means*, not just how "
            "it's stored), how many values are missing, how many are distinct, the statistics "
            "and the distribution. It's there to decide what to clean and how to transform. In "
            "big datasets, the distributions use a sample; the counts use every row.",
        ),
        "cleaning": Explanation(
            "Cleaning",
            "Cleaning fixes problems that don't depend on any sample: removing duplicates, "
            "converting types, dropping identifiers and useless columns. Since it learns nothing "
            "from the data, it can happen before the split. Each operation is recorded in the "
            "lineage, with the before and after.",
        ),
        "task": Explanation(
            "Task",
            "Choosing the target is choosing the question: what do you want to predict? A target "
            "with classes (text, boolean, an integer with few values) calls for classification; "
            "a continuous number calls for regression; without a target, only unsupervised tasks "
            "remain. The system suggests, but you decide.",
        ),
        "supervised": Explanation(
            "Supervised learning",
            "The model learns from examples that are already answered: each row brings the "
            "features and the **target**, the right answer. Training is adjusting the model to "
            "get the training target right; evaluating is seeing whether it gets rows it has "
            "never seen right. **Classification** predicts a class (survived or not, a flower's "
            "species); **regression** predicts a number (the price of a house).",
        ),
        "unsupervised": Explanation(
            "Unsupervised learning",
            "There's no target: the model looks for the structure of the data on its own. "
            "**Clustering** gathers similar rows into groups; **anomaly detection** points out "
            "the ones that don't look like the others; **dimensionality reduction** sums up many "
            "columns in a few, so they can be seen in a chart. As there's no right answer, it's "
            "evaluated by the shape of the groups, or by comparing them with a reference column "
            "the model didn't see.",
        ),
        "split": Explanation(
            "Split",
            "The model learns from the **training** data and is evaluated on the **test** data, "
            "on rows it has never seen: it's the only honest way to know whether it generalises "
            "or just memorised. The **stratified** split keeps the share of each class in every "
            "part (the target's classes or, in tasks without a target, the reference column's); "
            "the **time** split trains on the past and tests on the future. The seed makes the "
            "split repeatable.",
        ),
        "transform": Explanation(
            "Transformation",
            "Models need numbers with no gaps and on similar scales. Imputation fills in the "
            "nulls, encoding swaps categories for 0/1 columns, and scaling puts the columns in "
            "the same order of magnitude. All of this **learns from the training data only** "
            "(the median, the categories, the mean and the deviation) and is then applied to the "
            "validation and test data.",
        ),
        "leakage": Explanation(
            "Data leakage",
            "There's data leakage when the model, during training, gets information it wouldn't "
            "have when predicting: a mean computed with the test data, for example, or a column "
            "that only exists after the answer is known. The test result looks better than "
            "reality. Learning the transformations from the training data only avoids the most "
            "common case.",
        ),
        "projection": Explanation(
            "2D projection",
            "A chart only shows two dimensions. With two features, those are used; with more, "
            "PCA finds, from the training data only, the two directions in which the data varies "
            "most and projects every part onto them. It's a copy just for looking: the model "
            "uses every feature.",
        ),
        "output": Explanation(
            "Output",
            "At the end, the table becomes NumPy matrices of decimal numbers (float64), and the "
            "target's classes become the numbers 0, 1, 2… This is where pandas ends: from here "
            "on, the models only see numbers, but the names of the features and the classes go "
            "along with them.",
        ),
        # Como ler cada gráfico e cada tabela dos ecrãs de dados: a interface mostra-os por
        # baixo deles, e o painel "Porquê?" também os pode juntar.
        "upload": Explanation(
            "Loading a file",
            "It accepts CSV (separated by commas, semicolons or tabs, also compressed as `.gz`), "
            "Excel (`.xlsx`, `.xlsm` and the old `.xls`, the first sheet), JSON (a list of "
            "records, or one record per line as `.jsonl`; nested objects become columns like "
            "`address.city`) and Parquet (the column format of data tools, which already "
            "carries each column's type). The separator and the text encoding are detected on "
            "their own. The file is kept in the uploads folder (`data/uploads/`), which doesn't "
            "go to Git.",
        ),
        "correlation_matrix": Explanation(
            "Correlation matrix",
            "Each cell is the Pearson correlation between two numeric columns: 1 is a perfect "
            "linear relationship, -1 too, but the other way round (one goes up when the other "
            "goes down), and 0 is no linear relationship. Two highly correlated features tell "
            "the model almost the same thing. Correlation isn't causation.",
        ),
        "missing_map": Explanation(
            "Map of missing values",
            "Each row of the map is a row of the dataset (a selection spread across all of it) "
            "and each column is a column; the marked cells are missing values. Vertical stripes "
            "are columns with many nulls; horizontal ones, almost empty rows. Nulls that show up "
            "together in several columns usually have the same cause.",
        ),
        "class_balance": Explanation(
            "Class balance",
            "How many rows each class of the target has. The balance is the size of the least "
            "frequent class relative to the most frequent: 100% is perfectly balanced. With very "
            "imbalanced classes, a model that always answers the biggest class looks good on "
            "accuracy; the stratified split and metrics such as F1 start to matter more.",
        ),
        "group_means": Explanation(
            "Means by class",
            "The mean of each numeric feature in each class of the target. Big differences "
            "between the classes point to features that help separate them; nearly equal means, "
            "to features that say little on their own.",
        ),
        "lineage": Explanation(
            "Lineage",
            "Each row is an operation of the preparation, in the order it ran, with what "
            "changed: the rows before and after and the columns that came in, left or changed. "
            "It's with this list that the preparation is repeated, identically, another day or "
            "with new data.",
        ),
        "split_classes": Explanation(
            "Classes in each part",
            "The share of each class in the training, validation and test data. In the "
            "stratified split, the percentages are nearly the same in every part; in the random "
            "one, they can drift apart, especially with rare classes and little data.",
        ),
        "pca_loadings": Explanation(
            "Component loadings",
            "Each principal component is a sum of the features (already transformed), each with "
            "its own weight. A big weight, positive or negative, says the feature counts a lot "
            "in that direction; near zero, that it hardly counts.",
        ),
    }
)


def explain(topic: str) -> Explanation:
    """A explicação de um tema (``"split"``, ``"transform"``...).

    Raises:
        ComponentNotFoundError: Se o tema não existir.
    """
    try:
        return EXPLANATIONS[topic]
    except KeyError:
        raise ComponentNotFoundError("explanations", topic, EXPLANATIONS) from None
