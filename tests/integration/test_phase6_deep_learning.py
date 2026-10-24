"""Critério da fase 6: a MLP resolve as luas e os snapshots mostram a fronteira a curvar.

Pelos serviços: a rede chega a uma exatidão que a regressão logística não alcança, a
fronteira muda entre a primeira e a última fotografia, e a fronteira final é curva (uma
reta não a reproduz). Na interface: o ecrã Treinar mostra as camadas da rede.

As luas têm aqui ruído 0,1: com o ruído por omissão (0,2), as duas luas sobrepõem-se e
nenhum modelo passa muito dos 95% num teste de 40 pontos (a rede fica nos 94,6% em
média, contra 86% da regressão logística). Com menos ruído, "resolver" quer dizer
acertar quase tudo.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from ml_lab.core.services import (
    ExperimentConfig,
    ExperimentResult,
    Lab,
    ModelSpec,
    PreparationConfig,
    SourceSpec,
)

APP = Path(__file__).resolve().parents[2] / "ui" / "simple" / "app.py"
MOONS = SourceSpec("synthetic", {"generator": "moons", "seed": 0, "parameters": {"noise": 0.1}})


@pytest.fixture(scope="module")
def lab() -> Lab:
    return Lab.create()


def train(lab: Lab, model: str) -> ExperimentResult:
    return lab.experiments.run(
        ExperimentConfig(
            dataset=MOONS,
            preparation=PreparationConfig(target="target"),
            model=ModelSpec(name=model),
            seed=0,
        )
    )


def test_the_network_solves_the_moons_with_a_curving_boundary(lab: Lab) -> None:
    network, straight = train(lab, "mlp"), train(lab, "logistic_regression")

    animation = network.animation
    assert animation is not None
    first = np.asarray(animation.frames[0].values) > 0.5
    last = np.asarray(animation.frames[-1].values) > 0.5
    xs, ys = np.meshgrid(animation.xs, animation.ys)
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    # Se a fronteira final fosse uma reta, uma regressão logística reproduzia-a quase toda.
    line = LogisticRegression().fit(grid, last.ravel()).score(grid, last.ravel())

    accuracy = network.main_report.metric("accuracy").value
    assert accuracy is not None
    assert accuracy > 0.97
    assert accuracy > (straight.main_report.metric("accuracy").value or 0) + 0.05
    assert np.mean(first != last) > 0.1  # a fronteira mexeu-se durante o treino
    assert line < 0.9  # e acabou curva
    assert network.network is not None
    assert len(network.network.frames) > 10


def test_the_training_screen_shows_the_layers_of_the_network(lab: Lab) -> None:
    result = train(lab, "mlp")
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state["lab"] = lab
    app.session_state["result"] = result
    app.run()

    app.switch_page("pages/treinar.py").run()

    assert not app.exception
    assert "As camadas da rede" in [header.value for header in app.subheader]
    assert (
        len(app.get("plotly_chart")) >= 6
    )  # fronteira, loss, rede, gradientes, parados, ativações
    app.select_slider(key=f"treinar.network_step.{result.run_id}").set_value(
        result.network.frames[0].step if result.network else 0
    ).run()
    assert not app.exception
