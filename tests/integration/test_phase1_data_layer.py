"""Critério da fase 1: carregar o Titanic mostra o perfil correto, os avisos e a sugestão
"classificação binária" (o pandas isolado é confirmado em tests/architecture/).

Verifica-se nos serviços, que a interface usa através da API. O percurso na interface (do
início ao Titanic, explorado e preparado) está nos testes de ponta a ponta da web
(``ui/web/e2e/manual-path.spec.ts``).
"""

from ml_lab.core.services import ColumnKind, DatasetService, TaskType


def test_titanic_profile_warnings_and_task_suggestion() -> None:
    service = DatasetService()
    titanic = service.load_sample("titanic")

    profile = service.profile(titanic.id, target="Survived")
    suggestion = service.suggest_task(titanic.id, "Survived")

    assert (profile.n_rows, profile.n_columns) == (891, 12)
    assert profile.column("Age").n_missing == 177
    assert profile.column("Cabin").n_missing == 687
    assert profile.column("Embarked").n_missing == 2
    assert profile.column("PassengerId").kind is ColumnKind.IDENTIFIER
    assert profile.column("Survived").kind is ColumnKind.BOOLEAN
    assert profile.column("Sex").kind is ColumnKind.CATEGORICAL
    assert profile.column("Name").kind is ColumnKind.TEXT
    assert profile.class_balance is not None
    assert [(c.value, c.count) for c in profile.class_balance.classes] == [("0", 549), ("1", 342)]
    warnings = {(n.code, n.subject) for n in profile.all_notices if n.level == "warning"}
    assert {
        ("identifier_column", "PassengerId"),
        ("missing_values", "Age"),
        ("missing_values", "Cabin"),
    } <= warnings
    assert suggestion.task_type is TaskType.BINARY_CLASSIFICATION
