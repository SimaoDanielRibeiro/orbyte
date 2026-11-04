"""Avaliador de regressão: métricas de erro e comparação com prever sempre a média.

A referência mais simples de uma regressão é ignorar as features e prever sempre a
média do alvo: é o que dá R² = 0. Um modelo com R² negativo é pior do que isso.
Chegou na fase 3, com a regressão linear didática; a fase 4 junta-lhe os resíduos.
"""

from typing import Any, Final, override

import numpy as np

from ml_lab.core.evaluation.base import EVALUATORS, Evaluator, measure, undefined_notices
from ml_lab.core.evaluation.metrics.base import MetricInputs
from ml_lab.core.evaluation.report import EvaluationReport
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Features, LabModel, Target
from ml_lab.formatting import format_number

DISPLAY_ORDER: Final = ("r2", "rmse", "mae", "mse")


@EVALUATORS.register("regression")
class RegressionEvaluator(Evaluator):
    """Avalia modelos de regressão."""

    task_types = frozenset({TaskType.REGRESSION})

    @override
    def evaluate(
        self,
        model: LabModel[Any],
        X: Features,
        y: Target | None,
        *,
        partition: Partition,
        task_type: TaskType,
        class_labels: tuple[str, ...] = (),
    ) -> EvaluationReport:
        if y is None:
            raise ModelError("Evaluating a regression needs the target.")
        y_true = y.astype(np.float64)
        y_pred = np.asarray(model.predict(X), dtype=np.float64)
        results, undefined = measure(task_type, MetricInputs(y_true, y_pred), order=DISPLAY_ORDER)
        notices: list[Notice] = []
        r2 = next((m.value for m in results if m.name == "r2"), None)
        where = f"on the {partition.label.lower()} data"
        baseline_error = float(np.mean(np.abs(y_true - y_true.mean())))
        if r2 is not None and r2 < 0:
            notices.append(
                Notice.warning(
                    "below_baseline",
                    f"The R² {where} is {format_number(r2, 3)}: the model is worse than always "
                    "predicting the mean. Check the features and the preparation.",
                )
            )
        elif r2 is not None:
            notices.append(
                Notice.info(
                    "baseline",
                    f"For comparison: always predicting the mean would give R² = 0 and a mean "
                    f"absolute error of {format_number(baseline_error)} {where}; the model "
                    f"explains {format_number(r2 * 100, 1)}% of the target's variation.",
                )
            )
        notices += undefined_notices(
            undefined, partition, "there are too few rows, or the target never changes"
        )
        return EvaluationReport(
            partition=partition,
            task_type=task_type,
            n_samples=len(y_true),
            metrics=results,
            notices=tuple(notices),
        )
