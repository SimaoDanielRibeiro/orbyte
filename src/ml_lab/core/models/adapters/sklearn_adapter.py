"""Adaptador do contrato LabModel para os estimadores do scikit-learn.

Implementa o contrato uma só vez (regra 3: "adaptadores evitam repetição"). Cada
modelo concreto só diz qual é o estimador, com que hiperparâmetros, e o que mostrar
do seu estado.

Os avisos que o scikit-learn emite durante o treino (por exemplo, "o otimizador não
convergiu") não se perdem: tornam-se avisos didáticos do modelo, com uma explicação
do que fazer.
"""

from abc import abstractmethod
from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

from ml_lab.core.foundation.notices import Notice
from ml_lab.core.models.base import Features, LabModel, Target, capture_warnings
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext


class SklearnModel[P: Hyperparameters](LabModel[P]):
    """Base dos modelos que envolvem um estimador do scikit-learn."""

    estimator_: Any

    @abstractmethod
    def build_estimator(self) -> Any:
        """Cria o estimador do scikit-learn com os hiperparâmetros e a seed do modelo."""

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        estimator = self.build_estimator()
        _, caught = capture_warnings(lambda: estimator.fit(X, y))
        self.estimator_ = estimator
        for warning in caught:
            self._note(self._explain_warning(warning.message))

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        return np.asarray(self.estimator_.predict(X))

    @override
    def _predict_proba(self, X: Features) -> NDArray[np.float64]:
        return np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)

    @override
    def classes(self) -> NDArray[np.int64] | None:
        classes = getattr(getattr(self, "estimator_", None), "classes_", None)
        return None if classes is None else np.asarray(classes, dtype=np.int64)

    def _explain_warning(self, message: Warning | str) -> Notice:
        if isinstance(message, ConvergenceWarning):
            return Notice.warning(
                "convergence",
                f"O otimizador de '{self.info.display_name}' parou no limite de iterações sem "
                "convergir: os coeficientes ainda podiam melhorar. Aumenta o máximo de "
                "iterações ou escala as features (padronização).",
            )
        return Notice.info("library_warning", f"Aviso do scikit-learn: {message}")
