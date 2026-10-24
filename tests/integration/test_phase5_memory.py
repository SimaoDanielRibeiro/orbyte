"""Critério da fase 5: fechar a app, reabrir e comparar dois runs.

Pelos serviços: um ``Lab`` treina dois runs e guarda um; outro ``Lab`` com a mesma base de
dados (como reabrir a app) vê os dois no histórico, compara-os e reabre o guardado com o
modelo, a ponto de os gráficos do Avaliar voltarem a funcionar. Na interface: o botão
"Guardar experiência" da barra lateral e o ecrã Experiências, que compara e abre runs.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml_lab.config import Settings, get_settings
from ml_lab.core.services import (
    ExperimentConfig,
    ExperimentResult,
    Lab,
    ModelSpec,
    PreparationConfig,
    SourceSpec,
)

APP = Path(__file__).resolve().parents[2] / "ui" / "simple" / "app.py"
IRIS = SourceSpec("sample", {"name": "iris"})


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Uma base de dados e uma pasta de artefactos vazias, só deste teste (também para a app)."""
    monkeypatch.setenv("ML_LAB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ML_LAB_DATABASE_PATH", str(tmp_path / "artifacts" / "ml_lab.db"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def train(lab: Lab, c: float) -> ExperimentResult:
    lab.datasets.load(IRIS)
    return lab.experiments.run(
        ExperimentConfig(
            dataset=IRIS,
            preparation=PreparationConfig(target="species"),
            model=ModelSpec(name="logistic_regression", hyperparameters={"C": c}),
            seed=42,
        )
    )


def test_closing_reopening_and_comparing_two_runs(fresh: Settings) -> None:
    before = Lab.create(fresh)
    strong, weak = train(before, 10.0), train(before, 0.01)
    before.history.save(strong.run_id, name="C alto")

    after = Lab.create(fresh)  # fechar e reabrir a app: só fica o que está no disco
    runs = {entry.run_id: entry for entry in after.history.runs()}
    comparison = after.history.compare([strong.run_id, weak.run_id])
    reopened = after.history.open(strong.run_id)

    assert set(runs) == {strong.run_id, weak.run_id}
    assert (runs[strong.run_id].name, runs[strong.run_id].saved) == ("C alto", True)
    assert not runs[weak.run_id].saved
    assert [difference.label for difference in comparison.differences] == [
        "Hiperparâmetro · C (inverso da regularização)"
    ]
    accuracy = next(metric for metric in comparison.metrics if metric.name == "accuracy")
    assert accuracy.values == (
        strong.main_report.metric("accuracy").value,
        weak.main_report.metric("accuracy").value,
    )
    assert reopened == strong
    assert after.analysis.charts(strong.run_id).roc  # o modelo voltou do ficheiro


def test_the_interface_saves_compares_and_opens_runs(fresh: Settings) -> None:
    lab = Lab.create(fresh)
    strong, weak = train(lab, 10.0), train(lab, 0.01)
    app = AppTest.from_file(APP, default_timeout=120)
    # O AppTest perde a barra lateral quando a app volta a correr: o estado da sessão (os
    # serviços e o último resultado, como depois de treinar) vai antes da primeira corrida.
    app.session_state["lab"] = lab
    app.session_state["result"] = strong
    app.run()

    app.button(key="sidebar.save").click().run()
    app.switch_page("pages/experiencias.py").run()
    app.multiselect(key="experiencias.compare").set_value([strong.run_id, weak.run_id]).run()

    assert not app.exception
    assert lab.history.entry(strong.run_id).saved
    assert len(app.dataframe[0].value) == 2  # o histórico
    assert len(app.dataframe) == 3  # mais as métricas lado a lado e o que muda
    assert len(app.get("plotly_chart")) >= 1

    app.selectbox(key="experiencias.run").set_value(weak.run_id).run()
    app.button(key="experiencias.open").click().run()

    assert not app.exception
    assert app.session_state["result"].run_id == weak.run_id
