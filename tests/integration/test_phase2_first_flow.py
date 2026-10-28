"""Critério da fase 2: no Iris, escolher o modelo, treinar e ver as métricas.

Verifica-se nos serviços, o fluxo que a interface usa através da API. Na interface, o mesmo
percurso (escolher o modelo, treinar e abrir cada métrica) está nos testes de ponta a ponta da
web (``ui/web/e2e/training.spec.ts``).
"""

from ml_lab.core.services import (
    ExperimentConfig,
    Lab,
    ModelSpec,
    Partition,
    PreparationConfig,
    SplitConfig,
    SplitStrategy,
    TaskType,
)


def test_logistic_regression_learns_the_iris() -> None:
    lab = Lab.create()
    iris = lab.datasets.load_sample("iris")
    preparation = PreparationConfig(target="species", split=SplitConfig(SplitStrategy.STRATIFIED))

    cards = lab.catalog.models(lab.datasets.suggest_task(iris.id, "species").task_type)
    result = lab.experiments.run(
        ExperimentConfig(
            dataset=iris.source,
            preparation=preparation,
            model=ModelSpec(name="logistic_regression"),
            seed=42,
        )
    )

    assert "logistic_regression" in [card.name for card in cards]
    assert result.task_type is TaskType.MULTICLASS_CLASSIFICATION
    test = result.reports[Partition.TEST]
    accuracy = test.metric("accuracy").value
    assert accuracy is not None
    assert accuracy >= 0.9
    assert test.confusion_matrix is not None
    assert sum(map(sum, test.confusion_matrix.counts)) == 30
