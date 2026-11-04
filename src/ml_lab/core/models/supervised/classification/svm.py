"""Máquina de vetores de suporte (SVM) de classificação (scikit-learn).

Procura a fronteira com a maior margem possível: a faixa mais larga sem pontos entre as
classes. Só os pontos na margem (ou do lado errado) contam: são os vetores de suporte.
Com um kernel, a fronteira pode curvar-se sem nunca calcular novas features.

O SVM dá distâncias à fronteira, não probabilidades. Para as ter, calibra-se com o método
de Platt: uma regressão logística sobre as distâncias, ajustada com validação cruzada
(``CalibratedClassifierCV``, como o scikit-learn 1.9 recomenda). O SVM final é treinado
com todos os dados, e a classe prevista é a mais provável. Ficha didática:
``docs/algorithms/svm.md``.
"""

from typing import Any, Final, override

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.models.supervised.kernels import KernelParams
from ml_lab.core.training.context import TrainingContext

CALIBRATION_FOLDS: Final = 5
"""Folds da validação cruzada que calibra as probabilidades (menos, se uma classe for rara)."""


@MODELS.register("svm")
class Svm(SklearnModel[KernelParams]):
    """SVM de classificação do scikit-learn, com probabilidades calibradas."""

    info = ModelInfo(
        name="svm",
        display_name="Support vector machine (SVM)",
        summary=(
            "The boundary with the widest margin between the classes; with the RBF kernel, it "
            "curves around the data. Only the points near the boundary count."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.KERNEL,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=True,
        hyperparameters=KernelParams,
        doc="svm.md",
        keeps_training_rows=True,
    )

    def __init__(self, hyperparameters: KernelParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._folds = CALIBRATION_FOLDS

    @override
    def build_estimator(self) -> CalibratedClassifierCV:
        params = self.hyperparameters
        svm = SVC(C=params.C, kernel=params.kernel, gamma=params.gamma)
        return CalibratedClassifierCV(svm, method="sigmoid", cv=self._folds, ensemble=False)

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if y is None:
            raise ModelError("The SVM needs the target to classify.")
        rarest = int(np.unique(y, return_counts=True)[1].min())
        if rarest < 2:
            raise ModelError(
                "To calibrate the probabilities, each class needs at least two training examples."
            )
        self._folds = min(CALIBRATION_FOLDS, rarest)
        super()._fit(X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        svm = self.estimator_.calibrated_classifiers_[0].estimator
        return {
            "classes": self.estimator_.classes_.tolist(),
            "n_support": svm.n_support_.tolist(),
            "support_indices": svm.support_.tolist(),
        }
