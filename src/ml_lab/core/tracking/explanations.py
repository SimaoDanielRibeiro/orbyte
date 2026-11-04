"""Textos do painel "Porquê?" dos ecrãs Histórico e Guardar.

Registar, comparar e reproduzir runs; e guardar um modelo num pacote, levá-lo e voltar a
carregá-lo (ADR 0039).
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

TRACKING_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "experiment_tracking": Explanation(
            "Experiment tracking",
            "Every training run that finishes is recorded on its own: the configuration, the "
            "seed, the metrics, the loss curve and what the model learned. That's enough to "
            "compare runs and to repeat them. **Saving** a run adds the trained model and the "
            "step-by-step training (the animation and, for a neural network, the layers at each "
            "snapshot), which weigh more: a saved run opens again with everything, even after "
            "you close the app.",
        ),
        "comparing_runs": Explanation(
            "Comparing runs",
            "A comparison only teaches you something if the runs differ in one choice at a "
            "time: the same dataset, the same preparation and the same split, and only one "
            "hyperparameter different. If two things change at once, you can't tell which one "
            "made the difference. That's why the comparison shows only the choices that change.",
        ),
        "run_history": Explanation(
            "How to read the history",
            "The runs go from the most recent to the oldest, each with its main metric on the "
            "part that sums it up (the test, almost always). The saved ones open again with the "
            "model and the step-by-step training; the others, with everything but that: the "
            "metrics, the configuration and what the model learned, but without the charts "
            "that need the model or the Explain screen.",
        ),
        "reproducibility": Explanation(
            "Reproducibility",
            "A result only counts if it can be repeated. In Orbyte, the root seed decides all "
            "the others (the split's, the model's, the folds'), so the same configuration with "
            "the same seed gives exactly the same model and the same metrics. **Reproducing** a "
            "run trains it again that way; if the metrics change, something isn't "
            "deterministic, or the data has changed.",
        ),
        "model_package": Explanation(
            "The model package",
            "A model alone isn't enough to predict: it learned from the data **after** the "
            "preparation. A new row (a 22-year-old woman) has to go through the same "
            "transformations, with the values learned in training: the same median to fill in "
            "the missing age, the same categories, the same mean and standard deviation for "
            "scaling. That's why the package carries the preparation with the model. If the "
            "preparation changed between training and use, the model would get numbers it had "
            "never seen and be wrong without warning.",
        ),
        "model_card": Explanation(
            "The model card",
            "Whoever receives a model needs to know, before trusting it: what it predicts and "
            "from what, what data it learned from, how it did on the test (and compared with "
            "the simplest guess), and where it shouldn't be used. The most important limit is "
            "the ranges it has seen: outside them, the prediction is an **extrapolation**, a "
            "guess with no data behind it. The card is the package's README.md, written for "
            "people.",
        ),
        "pickle_safety": Explanation(
            "Why opening a model asks for confirmation",
            "The model goes in a joblib file, which is a Python *pickle*: opening it recreates "
            "the objects inside, and that can run code. A package made in this Orbyte carries "
            "its signature and opens straight away; one made elsewhere can carry its maker's "
            "code. That's why the card, the evaluation and the preparation (which are text) can "
            "be read at once, and the model only opens after you confirm you trust whoever gave "
            "you the file. A package changed after it was made never opens the model.",
        ),
        "onnx_format": Explanation(
            "ONNX",
            "ONNX (Open Neural Network Exchange) is an open format for exchanging trained models "
            "between programs. It describes the model as a graph of simple operations (add, "
            "multiply, compare), which ONNX Runtime runs in Python, C#, Java, JavaScript and "
            "other languages, without Orbyte or scikit-learn. The package's `model.onnx` joins "
            "the preparation and the model: it takes the columns as they come and returns the "
            "prediction. Before it goes into the package, it runs on the example rows and has to "
            "give the same predictions as the lab; if it doesn't, it's left out. It isn't a "
            "pickle, so it runs nobody's code.",
        ),
        "memorized_rows": Explanation(
            "Models that keep the data",
            "Some models don't sum up the data in weights: they keep it. KNN keeps every "
            "training row (it compares each new point with them), an SVM keeps the support "
            "vectors (the rows on the boundary) and DBSCAN the core points. Whoever has one of "
            "these models can see those rows, already transformed. Before sharing a model "
            "trained on personal data, think about who will have it.",
        ),
        "package_versions": Explanation(
            "Why the versions are recorded",
            "The manifest says which versions of Python, NumPy, scikit-learn and PyTorch made "
            "the package. A pickle depends on them: on another version, the model may not open, "
            "or may predict differently without raising an error. With the versions in view, "
            "you know what to install to repeat it; it's reproducibility extended beyond the "
            "seed.",
        ),
    }
)
"""Os temas do registo de experiências e dos pacotes."""
