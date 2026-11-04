"""Adaptador do contrato LabModel para os estimadores do scikit-learn.

Implementa o contrato uma só vez (regra 3: "adaptadores evitam repetição"). Cada
modelo concreto só diz qual é o estimador, com que hiperparâmetros, e o que mostrar
do seu estado.

Os avisos que o scikit-learn emite durante o treino (por exemplo, "o otimizador não
convergiu") não se perdem: tornam-se avisos didáticos do modelo, com uma explicação
do que fazer.

Alguns estimadores somam em paralelo (OpenMP), e a ordem das parcelas muda de uma corrida para
a outra: em vírgula flutuante, a soma muda no último dígito, e a mesma seed deixaria de dar
exatamente o mesmo resultado. Esses modelos dizem-no (``single_thread``) e treinam com uma só
thread.
"""

import threading
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, Final, override

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import Features, LabModel, Target, capture_warnings
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext

_ONE_THREAD: Final = threading.Lock()
"""O limite de threads é do processo todo: dois treinos ao mesmo tempo esperam um pelo outro,
para nenhum repor o limite enquanto o outro ainda conta com ele."""


@contextmanager
def _threads(single: bool) -> Iterator[None]:
    """Uma só thread de OpenMP enquanto dura o bloco, se ``single``; senão, as que houver."""
    if not single:
        yield
        return
    with _ONE_THREAD, threadpool_limits(limits=1, user_api="openmp"):
        yield


class SklearnModel[P: Hyperparameters](LabModel[P]):
    """Base dos modelos que envolvem um estimador do scikit-learn."""

    single_thread: ClassVar[bool] = False
    """Se treina com uma só thread: o estimador soma em paralelo, numa ordem que muda."""

    estimator_: Any

    @abstractmethod
    def build_estimator(self) -> Any:
        """Cria o estimador do scikit-learn com os hiperparâmetros e a seed do modelo."""

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self.estimator_ = self._fit_estimator(self.build_estimator(), X, y)

    def _fit_estimator(
        self,
        estimator: Any,
        X: Features,
        y: Target | None,
        *,
        ignore: Iterable[type[Warning]] = (),
    ) -> Any:
        """Treina ``estimator`` e transforma os avisos do scikit-learn em avisos do modelo.

        Args:
            estimator: O estimador.
            X: As features.
            y: O alvo (``None`` nas tarefas sem alvo).
            ignore: Avisos que não interessam (por exemplo, o de não convergir quando se
                treina de propósito um passo de cada vez).
        """
        ignored = tuple(ignore)
        with _threads(self.single_thread):
            _, caught = capture_warnings(lambda: estimator.fit(X, y))
        for warning in caught:
            if not ignored or not issubclass(warning.category, ignored):
                self._note(self._explain_warning(warning.message))
        return estimator

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        predictions = np.asarray(self.estimator_.predict(X))
        if TaskType.ANOMALY_DETECTION in self.info.task_types:
            flags: NDArray[np.int64] = (predictions == -1).astype(np.int64)  # marcadas com -1
            return flags
        return predictions

    @override
    def _predict_proba(self, X: Features) -> NDArray[np.float64]:
        return np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)

    @override
    def _transform(self, X: Features) -> NDArray[np.float64]:
        return np.asarray(self.estimator_.transform(X), dtype=np.float64)

    @override
    def _anomaly_scores(self, X: Features) -> NDArray[np.float64]:
        # O scikit-learn mede "quão normal" é cada ponto (maior = mais normal): a pontuação
        # de anomalia é o contrário.
        return -np.asarray(self.estimator_.score_samples(X), dtype=np.float64)

    @override
    def classes(self) -> NDArray[np.int64] | None:
        classes = getattr(getattr(self, "estimator_", None), "classes_", None)
        return None if classes is None else np.asarray(classes, dtype=np.int64)

    def _explain_warning(self, message: Warning | str) -> Notice:
        if isinstance(message, ConvergenceWarning):
            return Notice.warning(
                "convergence",
                f"The optimiser of '{self.info.display_name}' stopped at the iteration limit "
                "without converging: the coefficients could still improve. Raise the maximum "
                "number of iterations or scale the features (standardisation).",
            )
        return Notice.info("library_warning", f"scikit-learn warning: {message}")
