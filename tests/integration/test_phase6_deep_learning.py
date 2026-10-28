"""Critério da fase 6: a MLP resolve as luas e os snapshots mostram a fronteira a curvar.

Pelos serviços: a rede chega a uma exatidão que a regressão logística não alcança, a
fronteira muda entre a primeira e a última fotografia, e a fronteira final é curva (uma
reta não a reproduz). As camadas da rede na interface estão nos testes de ponta a ponta da
web (a página da rede neuronal e o ecrã Treinar e avaliar).

As luas têm aqui ruído 0,1: com o ruído por omissão (0,2), as duas luas sobrepõem-se e
nenhum modelo passa muito dos 95% num teste de 40 pontos (a rede fica nos 94,6% em
média, contra 86% da regressão logística). Com menos ruído, "resolver" quer dizer
acertar quase tudo.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml_lab.core.services import (
    ExperimentConfig,
    ExperimentResult,
    Lab,
    ModelSpec,
    PreparationConfig,
    SourceSpec,
)

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
