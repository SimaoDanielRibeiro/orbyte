"""Avaliador de redução de dimensionalidade: o que a projeção guardou dos vizinhos.

A pergunta é se o que fica junto na projeção estava junto nos dados. Um modelo que não
sabe projetar pontos novos (o t-SNE) só tem relatório do treino.
"""

from typing import Any, override

import numpy as np

from ml_lab.core.evaluation.base import EVALUATORS, Evaluator, measure, undefined_notices
from ml_lab.core.evaluation.metrics.base import MetricInputs
from ml_lab.core.evaluation.report import EvaluationReport
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Features, LabModel, Target


@EVALUATORS.register("dimensionality")
class DimensionalityEvaluator(Evaluator):
    """Avalia modelos de redução de dimensionalidade."""

    task_types = frozenset({TaskType.DIMENSIONALITY_REDUCTION})

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
        if not model.info.supports_new_data and partition is not Partition.TRAIN:
            return EvaluationReport(
                partition=partition,
                task_type=task_type,
                n_samples=len(X),
                metrics=(),
                notices=(
                    Notice.info(
                        "train_only",
                        f"'{model.info.display_name}' só coloca no mapa os pontos com que "
                        f"treinou: não há relatório do {partition.label.lower()}.",
                    ),
                ),
            )
        embedding = np.asarray(model.transform(X), dtype=np.float64)
        results, undefined = measure(task_type, MetricInputs(X=X, embedding=embedding))
        return EvaluationReport(
            partition=partition,
            task_type=task_type,
            n_samples=len(X),
            metrics=results,
            notices=tuple(undefined_notices(undefined, partition, "há poucas linhas")),
        )
