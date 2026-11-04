"""Explicação de cada métrica: o que mede, como se lê e a fórmula.

A gaveta de cada métrica e o glossário das Métricas mostram estes textos, escritos uma só
vez, em inglês como todo o site (ADR 0043). Um teste de contrato confirma que todas as métricas
registadas têm explicação, com a fórmula e o significado de cada símbolo.

As fórmulas estão em LaTeX (a interface mostra-as com o KaTeX). Os símbolos que
uma métrica dá como termos, com os números de um run (``Metric.breakdown``), aparecem
escritos na fórmula exatamente da mesma maneira.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class FormulaSymbol:
    """Um símbolo da fórmula e o que quer dizer.

    Attributes:
        symbol: O símbolo, em LaTeX.
        meaning: O que é (pode ter fórmulas curtas entre ``$``).
    """

    symbol: str
    meaning: str


@dataclass(frozen=True, slots=True)
class MetricExplanation:
    """O que uma métrica mede, como se lê e como se calcula.

    Attributes:
        what: O que mede, numa frase.
        how_to_read: Como interpretar o valor, com os cuidados a ter.
        formula: A definição, em LaTeX.
        symbols: O que quer dizer cada símbolo da fórmula.
    """

    what: str
    how_to_read: str
    formula: str
    symbols: tuple[FormulaSymbol, ...]


_N = FormulaSymbol("n", "the number of examples evaluated")
_Y = FormulaSymbol("y_i", "the true value of example $i$")
_Y_HAT = FormulaSymbol(r"\hat{y}_i", "the model's prediction for example $i$")
_TRUE_POSITIVES = FormulaSymbol(
    r"\mathrm{TP}",
    "true positives: the examples of the class that the model said belong to it",
)
_FALSE_POSITIVES = FormulaSymbol(
    r"\mathrm{FP}",
    "false positives: the examples of other classes that the model said belong to this one",
)
_FALSE_NEGATIVES = FormulaSymbol(
    r"\mathrm{FN}",
    "false negatives: the examples of the class that the model gave to another one",
)


METRIC_EXPLANATIONS: Final[Mapping[str, MetricExplanation]] = MappingProxyType(
    {
        "accuracy": MetricExplanation(
            "The share of predictions that are right.",
            "From 0% to 100%. It's the most intuitive, but it misleads with imbalanced classes: "
            "if 95% of the examples belong to one class, always predicting that class scores 95% "
            "without learning anything.",
            formula=r"\text{accuracy} = \frac{\text{correct}}{n}",
            symbols=(
                FormulaSymbol(r"\text{correct}", "the number of right predictions"),
                _N,
            ),
        ),
        "balanced_accuracy": MetricExplanation(
            "The mean of each class's recall: every class weighs the same.",
            "From 0% to 100%. With two classes, 50% is what chance gets. If it's much lower than "
            "the accuracy, the model is failing on the rare classes.",
            formula=r"\text{balanced accuracy} = \frac{1}{K} \sum_{c=1}^{K} \text{recall}_c",
            symbols=(
                FormulaSymbol("K", "the number of classes in the data evaluated"),
                FormulaSymbol(
                    r"\text{recall}_c",
                    "the recall of class $c$: the share of its examples the model found",
                ),
            ),
        ),
        "precision": MetricExplanation(
            "Of the times the model said “it's this class”, how many it got right.",
            "From 0% to 100%. It matters when a false alarm is costly (marking a good email as "
            "spam, for example). With several classes, it's the mean over the classes.",
            formula=r"\text{precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}",
            symbols=(_TRUE_POSITIVES, _FALSE_POSITIVES),
        ),
        "recall": MetricExplanation(
            "Of the examples that were of this class, how many the model found.",
            "From 0% to 100%. It matters when missing a case is costly (a diagnosis, for "
            "example). It rises easily at the expense of precision: always saying “yes” "
            "scores 100%.",
            formula=r"\text{recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}",
            symbols=(_TRUE_POSITIVES, _FALSE_NEGATIVES),
        ),
        "f1": MetricExplanation(
            "The harmonic mean of precision and recall.",
            "From 0% to 100%. It's only high if both are high: it sums up the balance between "
            "raising no false alarms and letting no case slip through.",
            formula=r"F_1 = 2 \cdot \frac{P \cdot R}{P + R}",
            symbols=(
                FormulaSymbol("P", "the precision"),
                FormulaSymbol("R", "the recall"),
            ),
        ),
        "log_loss": MetricExplanation(
            "How far the predicted probabilities are from the right answer.",
            "Lower is better; 0 would be absolute, always right certainty. It punishes confident "
            "wrong predictions hard. It only exists for models that give probabilities.",
            formula=r"\text{log-loss} = \frac{1}{n} \sum_{i=1}^{n} -\log p_i",
            symbols=(
                FormulaSymbol(
                    "p_i", "the probability the model gave to the right class of example $i$"
                ),
                _N,
            ),
        ),
        "r2": MetricExplanation(
            "The share of the target's variation that the model explains.",
            "1 is perfect; 0 is the same as always predicting the mean; below 0, the model is "
            "worse than that. It has no units, so it compares models on different targets.",
            formula=r"R^2 = 1 - \frac{\mathrm{SS}_{\text{res}}}{\mathrm{SS}_{\text{tot}}}",
            symbols=(
                FormulaSymbol(
                    r"\mathrm{SS}_{\text{res}}",
                    "the sum of squared residuals: $\\sum_i (y_i - \\hat{y}_i)^2$",
                ),
                FormulaSymbol(
                    r"\mathrm{SS}_{\text{tot}}",
                    "the sum of squares around the mean: $\\sum_i (y_i - \\bar{y})^2$, the error "
                    "of always predicting the mean",
                ),
            ),
        ),
        "rmse": MetricExplanation(
            "The root mean squared error: a typical prediction error, in the target's units.",
            "Lower is better. As the errors are squared, a few big errors weigh a lot: compare it "
            "with the mean absolute error to see whether some predictions miss by far.",
            formula=r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}",
            symbols=(_Y, _Y_HAT, _N),
        ),
        "mae": MetricExplanation(
            "The mean absolute error: the average distance between the prediction and the true "
            "value, in the target's units.",
            "Lower is better. It reads directly: “on average, the model misses by this much”. "
            "Every error weighs the same, big or small.",
            formula=r"\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|",
            symbols=(_Y, _Y_HAT, _N),
        ),
        "mse": MetricExplanation(
            "The mean squared error: the mean of the squared errors, the loss that linear "
            "regression minimises.",
            "Lower is better, but its units are the target's squared, which makes it hard to "
            "read; to interpret it, use its root (RMSE).",
            formula=r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2",
            symbols=(_Y, _Y_HAT, _N),
        ),
        "silhouette": MetricExplanation(
            "For each point, how much closer it is to the points of its own group than to those "
            "of the neighbouring group, on average.",
            "From -1 to 1. Near 1: compact, well-separated groups; near 0: groups pressed against "
            "each other; negative: some points are in the wrong group. It's good for comparing "
            "different values of k.",
            formula=r"s = \frac{1}{n} \sum_{i=1}^{n} \frac{b_i - a_i}{\max(a_i, b_i)}",
            symbols=(
                FormulaSymbol("a_i", "the mean distance from point $i$ to the points of its group"),
                FormulaSymbol(
                    "b_i",
                    "the mean distance from point $i$ to the points of the nearest other group",
                ),
                FormulaSymbol("n", "the number of points evaluated (without the noise)"),
            ),
        ),
        "davies_bouldin": MetricExplanation(
            "How similar each group is to its nearest neighbour: the spread inside the groups "
            "over the distance between them.",
            "Lower is better; 0 would be infinitely compact, far-apart groups. Like the "
            "silhouette, it prefers round groups.",
            formula=(
                r"\mathrm{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} "
                r"\frac{s_i + s_j}{d_{ij}}"
            ),
            symbols=(
                FormulaSymbol("k", "the number of groups"),
                FormulaSymbol(
                    "s_i", "the mean distance from the points of group $i$ to its centre"
                ),
                FormulaSymbol("d_{ij}", "the distance between the centres of groups $i$ and $j$"),
            ),
        ),
        "calinski_harabasz": MetricExplanation(
            "The variance between the groups over the variance inside the groups.",
            "Higher is better, but it has no fixed scale: it only compares groupings of the same "
            "data (k = 3 against k = 4, for example).",
            formula=r"\mathrm{CH} = \frac{B / (k - 1)}{W / (n - k)}",
            symbols=(
                FormulaSymbol(
                    "B",
                    "the spread between the groups: for each group, its number of points times "
                    "the squared distance from its centre to the centre of all the points",
                ),
                FormulaSymbol(
                    "W",
                    "the spread inside the groups: the sum of the squared distances from each "
                    "point to the centre of its group",
                ),
                FormulaSymbol("k", "the number of groups"),
                FormulaSymbol("n", "the number of points"),
            ),
        ),
        "n_clusters": MetricExplanation(
            "How many groups ended up with points (the noise doesn't count).",
            "It's neither good nor bad: in k-means it's the k you chose; in DBSCAN, it's what the "
            "density of the data dictated.",
            formula=r"k = \left| \{ g_i : g_i \neq \text{noise} \} \right|",
            symbols=(FormulaSymbol("g_i", "the group point $i$ ended up in"),),
        ),
        "noise_ratio": MetricExplanation(
            "The share of points that ended up in no group (only DBSCAN has noise).",
            "A lot of noise (more than half) means a radius that's too small; no noise at all may "
            "mean a radius that's too big, with everything in a single group.",
            formula=r"\text{noise} = \frac{n_{\text{noise}}}{n}",
            symbols=(
                FormulaSymbol(r"n_{\text{noise}}", "the points that ended up in no group"),
                FormulaSymbol("n", "the number of points evaluated"),
            ),
        ),
        "adjusted_rand": MetricExplanation(
            "The agreement between the groups found and the reference column, minus what chance "
            "would give.",
            "1 is perfect agreement; 0 is the same as grouping at random. The number each group "
            "gets doesn't matter, only who ends up with whom.",
            formula=(
                r"\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}"
                r"{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]}"
            ),
            symbols=(
                FormulaSymbol(
                    r"\mathrm{RI}",
                    "the Rand index: the share of pairs of points on which the groups and the "
                    "reference agree (together in both, or apart in both)",
                ),
                FormulaSymbol(
                    r"\mathbb{E}[\mathrm{RI}]", "the value the index would have with random groups"
                ),
            ),
        ),
        "trustworthiness": MetricExplanation(
            "Whether each point's neighbours in the projection were really neighbours in the "
            "original data (the 5 nearest).",
            "From 0% to 100%. Near 100%: what's together in the chart is together in the data. It "
            "doesn't promise the opposite: points far apart in the chart can be neighbours in the "
            "data.",
            formula=(
                r"T = 1 - \frac{2}{n k (2n - 3k - 1)} \sum_{i=1}^{n} \sum_{j \in U_i} "
                r"(r_{ij} - k)"
            ),
            symbols=(
                FormulaSymbol("k", "the number of neighbours compared (5)"),
                FormulaSymbol(
                    "U_i",
                    "the points that are neighbours of $i$ in the projection, but weren't in the "
                    "original data",
                ),
                FormulaSymbol(
                    "r_{ij}", "the rank of $j$ among the neighbours of $i$ in the original data"
                ),
                FormulaSymbol("n", "the number of points"),
            ),
        ),
        "anomaly_rate": MetricExplanation(
            "The share of points flagged as anomalies.",
            "It's neither good nor bad: it stays near the expected share you chose in the "
            "hyperparameters (on the training data, by construction).",
            formula=r"\text{flagged} = \frac{n_{\text{flagged}}}{n}",
            symbols=(
                FormulaSymbol(
                    r"n_{\text{flagged}}", "the points the detector flagged as anomalies"
                ),
                FormulaSymbol("n", "the number of points evaluated"),
            ),
        ),
        "anomaly_precision": MetricExplanation(
            "Of the points flagged as anomalies, how many really were (according to the "
            "reference column).",
            "From 0% to 100%. Low means many false alarms: the threshold is too low, or the "
            "anomalies don't stand out in the features.",
            formula=r"\text{precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}",
            symbols=(
                FormulaSymbol(r"\mathrm{TP}", "the true anomalies that were flagged"),
                FormulaSymbol(r"\mathrm{FP}", "the normal points flagged as anomalies"),
            ),
        ),
        "anomaly_recall": MetricExplanation(
            "Of the true anomalies, how many were flagged.",
            "From 0% to 100%. It rises if you expect more anomalies (the threshold drops), at "
            "the expense of precision.",
            formula=r"\text{recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}",
            symbols=(
                FormulaSymbol(r"\mathrm{TP}", "the true anomalies that were flagged"),
                FormulaSymbol(r"\mathrm{FN}", "the true anomalies left unflagged"),
            ),
        ),
        "anomaly_roc_auc": MetricExplanation(
            "The probability that a true anomaly gets a higher score than a normal point.",
            "From 0.5 (chance) to 1 (perfect separation). It doesn't depend on the threshold: it "
            "says whether the score ranks the points well, even if the cut is in the wrong place.",
            formula=r"\text{AUC} = P\left(s_{\text{anomaly}} > s_{\text{normal}}\right)",
            symbols=(
                FormulaSymbol(
                    r"s_{\text{anomaly}}", "the score of a true anomaly, drawn at random"
                ),
                FormulaSymbol(r"s_{\text{normal}}", "the score of a normal point, drawn at random"),
            ),
        ),
        "roc_auc": MetricExplanation(
            "The probability that the model scores an example of the positive class higher than "
            "one of the negative class.",
            "From 0.5 (chance) to 1 (perfect separation). It doesn't depend on the decision "
            "threshold, so it compares models well. With several classes, it's the mean of each "
            "class against the others.",
            formula=r"\text{AUC} = \int_0^1 \mathrm{TPR} \; d\,\mathrm{FPR}",
            symbols=(
                FormulaSymbol(
                    r"\mathrm{TPR}",
                    "the true positive rate (the recall) at each decision threshold",
                ),
                FormulaSymbol(
                    r"\mathrm{FPR}",
                    "the false positive rate: the share of negatives given as positives",
                ),
            ),
        ),
    }
)
