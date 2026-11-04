"""Explicações didáticas da avaliação, da explicabilidade e do Laboratório (painel "Porquê?").

Como as das outras camadas, vivem no backend, escritas uma só vez (em inglês, como todo o site;
ADR 0043), e a interface pede-as pela API; ``services/explanations.py`` junta-as todas.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

EVALUATION_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "cross_validation": Explanation(
            "Cross-validation",
            "A single test can come out better or worse by luck (the rows that happened to land "
            "in it). Cross-validation splits the training data into k folds and trains k times, "
            "each evaluated on a different fold: the mean estimates the performance on new data "
            "and the standard deviation says how much it varies. The transformations learn "
            "again in each fold, so the held-out fold never influences the training.",
        ),
        "learning_curve": Explanation(
            "Learning curve",
            "The model trains on more and more rows, and the training score is compared with the "
            "evaluation. Curves far apart at the end: the model memorises (variance), and more "
            "data or more regularisation help. Curves close but weak: the model is too simple "
            "(bias). The evaluation still rising at the end: more data would still help.",
        ),
        "roc_curve": Explanation(
            "ROC and precision-recall curves",
            "A classifier with probabilities decides with a threshold (50% by default). Each "
            "threshold gives a mix of hits and false alarms; the ROC curve shows them all, and "
            "the area under it (AUC) sums them up in a number that doesn't depend on the "
            "threshold. With very imbalanced classes, the precision-recall curve is more honest.",
        ),
        "residuals": Explanation(
            "Residuals",
            "The residual is the error of each prediction: actual minus predicted. In a good "
            "model, the residuals look like noise around zero. A curve in the residuals means the "
            "model missed a curved relationship; a fan opening up means the error grows with the "
            "value (predicting the logarithm of the target may be worth it).",
        ),
        "silhouette": Explanation(
            "Silhouette",
            "For each point, it compares the mean distance to the points of its group with the "
            "mean distance to those of the nearest other group: near 1, the point sits well in "
            "its group; near 0, it's on the border; negative, it would be better in another. The "
            "profile shows every point of each group, in order: groups with many short or "
            "negative bars are poorly defined.",
        ),
        "reference": Explanation(
            "Reference column",
            "In tasks without a target, a target column chosen in Prepare doesn't go into the "
            "training: it stays as a reference, to compare what the model found with it at the "
            "end (the k-means groups with the species, the flagged anomalies with the true "
            "ones). It's an honest way of evaluating unsupervised learning.",
        ),
        "permutation_importance": Explanation(
            "Permutation importance",
            "To know how much the model depends on a feature, its values are shuffled in the "
            "test set and the drop in the metric is measured. It works with any model and "
            "measures on the test set, so it says what the model uses to generalise. With "
            "correlated features, shuffling one costs little: the other stands in for it.",
        ),
        "partial_dependence": Explanation(
            "Partial dependence",
            "It shows how the prediction changes, on average, when one feature changes and the "
            "others stay as they are. The thin lines (ICE) do the same for individual rows: if "
            "they're parallel, the effect is the same for all; if they cross, the feature "
            "interacts with others. Beware of combinations that don't exist in the data.",
        ),
        "local_explanation": Explanation(
            "Explaining one prediction",
            "In linear models, the prediction is the intercept plus the sum of weight × value of "
            "each feature: each term is the feature's exact contribution. In other models, it "
            "measures how much the prediction changes when each feature takes the training mean "
            "instead: an approximation, which ignores the interactions between features.",
        ),
        "bias_variance": Explanation(
            "Bias and variance",
            "A model that's too simple always makes the same kind of mistake (bias): it does "
            "badly on both training and test data. A model that's too flexible follows the noise "
            "of the training data (variance): it does great on training and badly on test. The "
            "best is in between, and you find it by looking at the test, never the training.",
        ),
        "learning_rate_demo": Explanation(
            "Learning rate",
            "The error of linear regression is a bowl with its bottom at the best weights. "
            "Gradient descent goes down the steepest side, in steps the size of the rate. Too "
            "small, it takes forever; too big, it jumps from one side of the bowl to the other; "
            "bigger still, it jumps higher and higher and blows up.",
        ),
        "knn_k": Explanation(
            "The k in KNN",
            "With k = 1, each training point decides its neighbourhood alone: it gets 100% of "
            "the training data right and draws islands around the noise. With a big k, the vote "
            "gathers points that are too far away and the boundary gets too smooth. The right k "
            "balances the two, and it's chosen on the test set (or with cross-validation).",
        ),
        "regularization": Explanation(
            "Regularisation",
            "Adding a cost for the size of the weights to the error forces the model to be "
            "simpler. In Ridge (L2) all the weights shrink; in Lasso (L1) the least useful ones "
            "reach exactly zero. Too much alpha also wipes out what matters; the best alpha is "
            "chosen on the test set.",
        ),
        "decision_threshold": Explanation(
            "Decision threshold",
            "A classifier with probabilities only decides when the probability is compared with "
            "a threshold. Lowering it flags more rows: more of the real positives are found "
            "(recall), but more of the flags are false alarms (precision). Raising it does the "
            "opposite. F1 balances the two. With imbalanced classes, accuracy hides all this: "
            "always answering the larger class already scores high. The threshold is chosen by "
            "the cost of each kind of mistake, on validation data, never on the test.",
        ),
        # Como ler os gráficos da avaliação: a interface mostra-os por baixo deles.
        "train_vs_test": Explanation(
            "Training and test",
            "The model learned from the training data; the test shows how it does on data it "
            "has never seen. A big gap in favour of the training data is the sign of "
            "overfitting.",
        ),
        "confusion_matrix": Explanation(
            "Confusion matrix",
            "Each row is the actual class and each column the predicted one: the diagonal holds "
            "the hits, and the rest are the model's confusions.",
        ),
        "reference_matrix": Explanation(
            "Groups and reference",
            "Each row is a class of the reference column and each column a group found: groups "
            "that match classes have almost everything in a single cell of the column.",
        ),
        "anomaly_matrix": Explanation(
            "Anomalies and reference",
            "Each row is what the point was (according to the reference column) and each column "
            "what the detector said: the diagonal holds the hits.",
        ),
        "roc_pr_chart": Explanation(
            "How to read the curves",
            "Each point of a curve is a different decision threshold. The ROC compares the hits "
            "with the false alarms (the diagonal is chance); the precision-recall curve shows "
            "the price of finding more cases. With several classes, there's one curve per class "
            "against the others.",
        ),
        "errors_map": Explanation(
            "Where the model is wrong",
            "The test points on the plane of the 2D projection from Prepare. Areas with many "
            "errors show where the model struggles most (often, where the classes mix).",
        ),
        "cluster_sizes": Explanation(
            "The size of the groups",
            "How many test points ended up in each group. Very uneven groups (one huge and "
            "several tiny) usually mean the wrong number of groups.",
        ),
        "dendrogram": Explanation(
            "Dendrogram",
            "Each “U” joins two groups, at the height of the distance at which they merged "
            "(only the last 30). The horizontal line is the cut that gives the chosen groups: a "
            "merge much higher than the ones below is a good place to cut.",
        ),
        "explained_variance": Explanation(
            "Explained variance",
            "How much of the data's variance each component keeps, in order. The sum is what "
            "the reduction kept; what's missing was lost in it.",
        ),
        "anomaly_scores": Explanation(
            "The anomaly scores",
            "The score of each test point. Above the threshold (the line), the detector flags "
            "the point as an anomaly. With a reference column, the colours are the true "
            "anomalies: the ones below the threshold escaped the detector, and the normal points "
            "above it are false alarms. Without it, the colours are what the detector decided.",
        ),
        "anomaly_roc": Explanation(
            "The score's ROC",
            "The ROC curve of the score against the true anomalies (from the reference "
            "column): it doesn't depend on the threshold.",
        ),
        # Como ler os gráficos do ecrã Explicar.
        "importance_bars": Explanation(
            "How to read the importance",
            "Each bar is how much the main metric drops on the test set when the feature is "
            "shuffled, and the whisker around its end is the spread between the repeats. Near "
            "zero, or negative: the model doesn't need that feature, or a correlated one stands "
            "in for it.",
        ),
        "dependence_curves": Explanation(
            "How to read the partial dependence",
            "The bold line is the mean prediction when every test row gets that value of the "
            "feature; the faint ones are a few rows, one by one. The ticks at the bottom show "
            "where there's data: far from them, the curve is pure extrapolation. The feature's "
            "values are on the scale after the preparation (standardised, if that's the case).",
        ),
        # Como ler o que um modelo aprendeu (a página de cada modelo e o fim de um treino).
        "impurity_importance": Explanation(
            "Impurity importance",
            "How much each feature reduced the impurity in the trees' splits, summed over all of "
            "them and normalised to add up to 1. It favours features with many different values: "
            "permutation importance (in Explain) is more honest.",
        ),
        "tree_diagram": Explanation(
            "How to read the tree",
            "Each node asks a feature a question (“x2 ≤ 0.26?”): the points that answer yes go "
            "left, the others go right. The leaves, at the bottom, give the prediction: the most "
            "frequent class among the training points that reach them, with the share it has. A "
            "pure leaf (100%) only has points of one class; many small, pure leaves are the sign "
            "of a tree that memorised the training data.",
        ),
    }
)
