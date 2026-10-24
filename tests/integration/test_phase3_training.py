"""Critério da fase 3: ver a fronteira da regressão logística didática a mover-se, com o
resultado final igual ao do scikit-learn.

Verifica-se nos serviços (as fotografias do treino e a comparação com o scikit-learn) e
na interface simples: Modelo → Treinar mostra a animação da fronteira.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ml_lab.core.services import (
    ExperimentConfig,
    ExperimentResult,
    Lab,
    ModelSpec,
    Partition,
    PreparationConfig,
    SourceSpec,
)

APP = Path(__file__).resolve().parents[2] / "ui" / "simple" / "app.py"
MOONS = SourceSpec(
    "synthetic", {"generator": "moons", "seed": 0, "parameters": {"n_samples": 400, "noise": 0.2}}
)


@pytest.fixture(scope="module")
def lab() -> Lab:
    return Lab.create()


def train(lab: Lab, model: str) -> ExperimentResult:
    return lab.experiments.run(
        ExperimentConfig(
            dataset=MOONS,
            preparation=PreparationConfig(target="target"),
            model=ModelSpec(name=model),
            seed=42,
        )
    )


def test_the_didactic_boundary_moves_during_training(lab: Lab) -> None:
    result = train(lab, "logistic_regression_gd")

    animation = result.animation
    assert animation is not None
    assert animation.animated
    assert (animation.dimensions, animation.value_kind) == (2, "probability")
    assert len(animation.frames) >= 10
    first, last = (np.array(frame.values) for frame in (animation.frames[0], animation.frames[-1]))
    assert np.abs(last - first).max() > 0.2  # as probabilidades mudam...
    assert np.mean((first >= 0.5) != (last >= 0.5)) > 0.01  # ...e a fronteira desloca-se
    losses = [frame.loss for frame in animation.frames if frame.loss is not None]
    assert len(losses) == len(animation.frames)
    assert losses[-1] < losses[0]


def test_the_final_result_equals_the_scikit_learn_one(lab: Lab) -> None:
    ours = train(lab, "logistic_regression_gd")
    theirs = train(lab, "logistic_regression")
    test = lab.experiments.artifacts(ours.run_id).dataset.splits[Partition.TEST].X
    our_model = lab.experiments.artifacts(ours.run_id).model
    their_model = lab.experiments.artifacts(theirs.run_id).model

    assert np.array_equal(our_model.predict(test), their_model.predict(test))
    assert np.allclose(our_model.predict_proba(test), their_model.predict_proba(test), atol=1e-4)
    assert np.allclose(
        ours.model_state["coefficients"], theirs.model_state["coefficients"], atol=1e-4
    )
    assert ours.reports[Partition.TEST].values() == pytest.approx(
        theirs.reports[Partition.TEST].values(), abs=1e-4
    )


def test_the_training_screen_replays_the_boundary() -> None:
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.selectbox(key="dados.sample").select("iris").run()
    app.button(key="dados.load").click().run()
    app.switch_page("pages/preparar.py").run()
    app.switch_page("pages/modelo.py").run()
    app.selectbox(key="modelo.multiclass_classification").select("logistic_regression_gd").run()

    app.button(key="modelo.train").click().run()
    assert app.title[0].value == "Treinar"  # o botão leva logo ao ecrã Treinar
    app.session_state["job"].wait(timeout=60)
    # O AppTest volta a correr a página que ele próprio abriu (o Modelo): muda-se à mão.
    app.switch_page("pages/treinar.py").run()

    assert not app.exception
    animation, loss = (json.loads(chart.proto.spec) for chart in app.get("plotly_chart"))
    assert len(animation["frames"]) > 1
    assert animation["layout"]["sliders"][0]["steps"]
    assert loss["data"][0]["name"] == "Loss (treino)"
