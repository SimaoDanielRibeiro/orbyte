"""Critério da fase 9: os agentes encontram um modelo para o Titanic sem tocar no teste.

Pelos serviços: o agente de dados prepara o Titanic e diz porquê; a triagem passa por uma
família de cada vez, com os valores por omissão; só os finalistas são afinados; tudo se mede
com validação cruzada nos folds do treino. O vencedor é o único run que usa o teste: fica no
registo em SQLite, e outro ``Lab`` (como reabrir a app) abre-o.

Na interface, o ecrã Agentes (o pedido, o fluxograma ao vivo, a pausa e o vencedor) está nos
testes de ponta a ponta da web (``ui/web/e2e/agents.spec.ts``).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.services import (
    AutoMLRequest,
    Lab,
    SearchPhase,
    SearchSettings,
    SourceSpec,
)

TITANIC = SourceSpec("sample", {"name": "titanic"})


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Uma base de dados e uma pasta de artefactos vazias, só deste teste."""
    monkeypatch.setenv("ML_LAB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ML_LAB_DATABASE_PATH", str(tmp_path / "artifacts" / "ml_lab.db"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def test_the_agents_find_a_model_for_the_titanic_without_touching_the_test(
    fresh: Settings,
) -> None:
    lab = Lab.create(fresh)
    request = AutoMLRequest(
        dataset=TITANIC,
        target="Survived",
        settings=SearchSettings(budget=15, folds=3, finalists=3),
        seed=42,
    )

    result = lab.automl.start(request).wait(timeout=300)

    # 1. O agente de dados preparou o Titanic, com a razão de cada decisão.
    titles = [decision.title for decision in result.plan.decisions]
    assert titles[:4] == ["Remover PassengerId", "Remover Name", "Remover Ticket", "Remover Cabin"]
    assert result.plan.objective.name == "accuracy"
    # 2. A triagem passou por todos os modelos de todas as famílias, com os valores por omissão.
    screening = [c for c in result.candidates if c.phase is SearchPhase.SCREENING]
    assert len(screening) == 9
    assert len({c.family for c in screening}) == 7
    assert all(not c.changes for c in screening)
    # 3. Só os três finalistas da triagem foram afinados.
    tuning = [c for c in result.candidates if c.phase is SearchPhase.TUNING]
    finalists = sorted(screening, key=lambda c: (-(c.mean or 0.0), c.std or 0.0, c.number))[:3]
    assert len(tuning) == 6
    assert {c.model for c in tuning} <= {c.model for c in finalists}
    # 4. O teste usou-se uma só vez: o registo tem um único run, o do vencedor.
    winner, run_id = result.winner_candidate, result.run_id
    assert winner is not None
    assert run_id is not None
    assert [entry.run_id for entry in lab.history.runs()] == [run_id]
    assert result.test is not None
    assert result.test.value is not None
    assert result.test.value > 0.75
    # 5. Na validação cruzada, o vencedor bate (ou empata) a referência.
    assert result.baseline is not None
    baseline = result.candidate(result.baseline)
    assert baseline.model == "logistic_regression"
    assert winner.mean is not None
    assert baseline.mean is not None
    assert winner.mean >= baseline.mean
    # 6. Fechar e reabrir a app: o vencedor está no disco e abre-se com o mesmo teste.
    after = Lab.create(fresh)
    entry = after.history.entry(run_id)
    assert entry.model_name == winner.model
    assert entry.name.startswith("Agentes · ")
    reopened = after.history.open(run_id)
    assert reopened.main_report.metric("accuracy").value == result.test.value
