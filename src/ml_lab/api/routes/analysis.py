"""Rotas do que se pergunta a um modelo depois do treino (a avaliação e o ecrã Explicar).

Só servem os runs que ainda estão em memória (os últimos desta sessão, ou um run guardado
que se reabriu do histórico): precisam do modelo treinado e dos dados.
"""

from typing import Annotated

from fastapi import Query

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.core.services import (
    CrossValidation,
    EvaluationCharts,
    LearningCurve,
    PartialDependence,
    PermutationImportance,
    RowExplanation,
)

router = ServiceRouter(prefix="/analysis/{run_id}", tags=["analysis"], responses=ERROR_RESPONSES)

Folds = Annotated[int | None, Query(ge=2, le=20, description="Quantos folds (por omissão, 5).")]


@router.get("/charts")
def charts(lab: LabDep, run_id: str) -> EvaluationCharts:
    """Os dados dos gráficos da avaliação (ROC, resíduos, silhueta…)."""
    return lab.analysis.charts(run_id)


@router.get("/cross-validation")
def cross_validation(lab: LabDep, run_id: str, k: Folds = None) -> CrossValidation:
    """A validação cruzada do modelo, com os mesmos hiperparâmetros e a mesma seed."""
    if k is None:  # o número de folds por omissão é o do serviço
        return lab.analysis.cross_validation(run_id)
    return lab.analysis.cross_validation(run_id, k)


@router.get("/learning-curve")
def learning_curve(lab: LabDep, run_id: str, k: Folds = None) -> LearningCurve:
    """A curva de aprendizagem, nos folds da validação cruzada."""
    if k is None:
        return lab.analysis.learning_curve(run_id)
    return lab.analysis.learning_curve(run_id, k)


@router.get("/permutation-importance")
def permutation_importance(lab: LabDep, run_id: str) -> PermutationImportance:
    """A importância por permutação de cada feature, no teste."""
    return lab.analysis.permutation_importance(run_id)


@router.get("/partial-dependence")
def partial_dependence(lab: LabDep, run_id: str, feature: str) -> PartialDependence:
    """A dependência parcial de uma feature (com as curvas ICE), no teste."""
    return lab.analysis.partial_dependence(run_id, feature)


@router.get("/test-rows")
def test_rows(lab: LabDep, run_id: str) -> int:
    """Quantas linhas de teste tem o run (para escolher uma previsão a explicar)."""
    return lab.analysis.test_rows(run_id)


@router.get("/test-rows/{row}/explanation")
def explain_row(lab: LabDep, run_id: str, row: int) -> RowExplanation:
    """O contributo de cada feature para a previsão de uma linha do teste."""
    return lab.analysis.explain_row(run_id, row)
