"""As demonstrações do ecrã Laboratório mostram o que dizem mostrar."""

import numpy as np
import pytest

from ml_lab.core.services import LaboratoryService


@pytest.fixture(scope="module")
def laboratory() -> LaboratoryService:
    return LaboratoryService()


def test_overfitting_the_test_error_goes_down_then_up(laboratory: LaboratoryService) -> None:
    demo = laboratory.overfitting(15)

    best = int(np.argmin(demo.test_errors))
    assert 0 < best < len(demo.degrees) - 1  # o melhor grau está no meio
    assert demo.train_errors[-1] < demo.train_errors[0]  # o treino só melhora
    assert demo.test_errors[-1] > demo.test_errors[best]
    assert [notice.code for notice in demo.notices] == ["overfitting"]
    assert [notice.code for notice in laboratory.overfitting(1).notices] == ["underfitting"]


@pytest.mark.parametrize(
    ("rate", "code"),
    [(0.01, "slow"), (0.3, "converging"), (1.2, "growing"), (2.5, "diverged")],
)
def test_the_learning_rate_decides_how_the_descent_goes(
    laboratory: LaboratoryService, rate: float, code: str
) -> None:
    demo = laboratory.learning_rate(rate)

    assert [notice.code for notice in demo.notices] == [code]
    assert demo.path.x[0] == 0.0  # os pesos começam a zero
    assert rate in [curve.rate for curve in demo.curves]


def test_the_converging_path_ends_at_the_bottom_of_the_bowl(
    laboratory: LaboratoryService,
) -> None:
    demo = laboratory.learning_rate(0.3)

    # Perto do fundo: ao longo do vale estreito (as features são correlacionadas), o passo
    # final ainda anda devagar.
    end = np.array([demo.path.x[-1], demo.path.y[-1]])
    assert np.allclose(end, demo.optimum, atol=0.05)


def test_knn_one_neighbour_memorises_and_the_best_k_is_in_the_middle(
    laboratory: LaboratoryService,
) -> None:
    demo = laboratory.knn(1)

    assert demo.train_accuracy[0] == pytest.approx(1.0)
    best = demo.ks[int(np.argmax(demo.test_accuracy))]
    assert 1 < best < demo.ks[-1]
    assert demo.boundary is not None
    assert [notice.code for notice in demo.notices] == ["k_small"]


def test_lasso_zeroes_the_useless_weights_and_ridge_does_not(
    laboratory: LaboratoryService,
) -> None:
    lasso = laboratory.regularization("lasso", 0.5)
    ridge = laboratory.regularization("ridge", 10.0)

    assert np.count_nonzero(np.abs(lasso.chosen[3:]) < 1e-10) >= 3  # x4…x8 não contam
    assert np.count_nonzero(np.abs(ridge.chosen) < 1e-10) == 0
    # Com alpha enorme, todos os pesos encolhem para perto de zero.
    assert max(abs(path[-1]) for path in ridge.paths) < 0.5
