"""Modelos do scikit-learn que se mostram a crescer: por profundidade, por árvores, por etapas.

O scikit-learn treina estes modelos de uma vez. Para se ver o treino passo a passo, o
modelo treina-se aos bocados, e cada bocado é um passo do ML Lab, com a sua fotografia:

- **por profundidade** (árvores de decisão): a árvore com profundidade 1, 2, 3… Como cada
  divisão escolhida não depende do limite de profundidade, a árvore de profundidade d é
  o início da árvore completa;
- **por árvores** (Random Forest): o ``warm_start`` junta árvores às que já existem, e o
  scikit-learn garante que o resultado é igual ao de treinar todas de uma vez;
- **por etapas** (boosting): treina tudo e depois percorre as etapas com as previsões
  "por etapa" (``staged_predict``): a etapa m é exatamente o modelo com as primeiras m
  árvores. Se o treino for parado na etapa m, volta a treinar só com m árvores, que são
  as mesmas (as seeds são tiradas pela mesma ordem).

Quando ninguém está a ver (``context.observed`` é falso), treinam de uma vez: o
resultado é o mesmo, mais depressa.
"""

import math
from collections.abc import Iterator, Mapping
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import Features, Target
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext

GROWTH_STEPS: Final = 20
"""Em quantos bocados cresce uma floresta (as árvores juntam-se aos grupos)."""


class GrowingSklearnModel[P: Hyperparameters](SklearnModel[P]):
    """Base dos modelos do scikit-learn que se mostram a crescer (ver o módulo)."""

    def _training_error(self, predictions: NDArray[Any], y: Target) -> float:
        """A loss de cada passo: o erro quadrático médio ou a proporção de erros."""
        if TaskType.REGRESSION in self.info.task_types:
            return float(np.mean((np.asarray(predictions, dtype=np.float64) - y) ** 2))
        return float(np.mean(np.asarray(predictions) != y))

    def _announce(
        self,
        step: int,
        loss: float,
        context: TrainingContext,
        *,
        metrics: Mapping[str, float] | None = None,
        probe: NDArray[np.float64] | None = None,
    ) -> None:
        """Anuncia o passo e, se o contexto a quiser, entrega a fotografia.

        Args:
            step: O passo.
            loss: A loss nesse passo.
            context: O contexto do treino.
            metrics: Métricas parciais.
            probe: As previsões na sonda, se já estiverem calculadas (nas etapas do
                boosting); senão, calcula-as o próprio modelo.
        """
        context.report_step(step, loss=loss, metrics=metrics)
        if not context.wants_snapshot(step):
            return
        if probe is None:
            context.report_snapshot(step, self.snapshot_state(context.probe))
        else:
            context.report_snapshot(step, {**self.state(), "probe": probe})

    def _grow_by_depth(
        self, estimator: Any, X: Features, y: Target | None, context: TrainingContext
    ) -> None:
        """Árvores: profundidade 1, 2, 3… até ao limite (ou até a árvore deixar de crescer)."""
        if not context.observed or y is None:
            self.estimator_ = self._fit_estimator(estimator, X, y)
            return
        limit = estimator.get_params()["max_depth"]
        depth = 1
        while True:
            grown = self._fit_estimator(clone(estimator).set_params(max_depth=depth), X, y)
            if depth > 1 and grown.get_depth() < depth:  # já não cresce: está completa
                self.estimator_ = grown
                return
            self.estimator_ = grown
            self._announce(depth, self._training_error(grown.predict(X), y), context)
            if (limit is not None and depth >= limit) or context.should_stop():
                return
            depth += 1

    def _grow_by_trees(
        self, estimator: Any, X: Features, y: Target | None, context: TrainingContext
    ) -> None:
        """Florestas: junta árvores aos grupos e anuncia o erro "fora do saco" (OOB)."""
        total = int(estimator.get_params()["n_estimators"])
        if not context.observed or y is None:
            self.estimator_ = self._fit_estimator(estimator, X, y)
            return
        parts = range(1, GROWTH_STEPS + 1)
        sizes = sorted({math.ceil(total * part / GROWTH_STEPS) for part in parts})
        growing = clone(estimator).set_params(warm_start=True, oob_score=True)
        for step, size in enumerate(sizes, start=1):
            growing.set_params(n_estimators=size)
            # Com poucas árvores, há linhas que nunca ficam fora do saco: o aviso não interessa.
            ignore = (UserWarning,) if size < total else ()
            self.estimator_ = self._fit_estimator(growing, X, y, ignore=ignore)
            oob = float(growing.oob_score_)
            self._announce(step, 1.0 - oob, context, metrics={"oob_score": oob})
            if context.should_stop():
                return

    def _replay_stages(
        self, estimator: Any, X: Features, y: Target | None, context: TrainingContext
    ) -> None:
        """Boosting: treina tudo e percorre as etapas; se for parado, fica com as primeiras."""
        self.estimator_ = self._fit_estimator(estimator, X, y)
        if not context.observed or y is None:
            return
        probe = context.probe
        for stage, (predictions, field) in enumerate(self._stages(X, probe), start=1):
            self._announce(stage, self._training_error(predictions, y), context, probe=field)
            if context.should_stop() and stage < self._stage_count():
                truncated = clone(estimator).set_params(n_estimators=stage)
                self.estimator_ = self._fit_estimator(truncated, X, y)
                return

    def _stages(
        self, X: Features, probe: NDArray[np.float64] | None
    ) -> Iterator[tuple[NDArray[Any], NDArray[np.float64] | None]]:
        """As previsões no treino e na sonda, etapa a etapa."""
        estimator = self.estimator_
        on_train = estimator.staged_predict(X)
        if probe is None:
            for predictions in on_train:
                yield predictions, None
            return
        if self.info.supports_proba:
            on_probe = estimator.staged_predict_proba(probe)
        else:
            on_probe = estimator.staged_predict(probe)
        for predictions, field in zip(on_train, on_probe, strict=True):
            yield predictions, np.asarray(field, dtype=np.float64)

    def _stage_count(self) -> int:
        return len(self.estimator_.estimators_)
