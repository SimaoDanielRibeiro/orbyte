"""As demonstrações do ecrã Laboratório mostram o que dizem mostrar."""

import numpy as np
import pytest

from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.services import (
    DemoControl,
    LaboratoryDemo,
    LaboratoryService,
    ThresholdDemo,
    explain,
)


@pytest.fixture(scope="module")
def laboratory() -> LaboratoryService:
    return LaboratoryService()


def controls(laboratory: LaboratoryService) -> dict[str, DemoControl]:
    """O controlo de cada demonstração (e de cada versão), pelo nome."""
    found: dict[str, DemoControl] = {}
    for card in laboratory.demos():
        if card.control is not None:
            found[card.name] = card.control
        for variant in card.variants:
            found[f"{card.name}.{variant.name}"] = variant.control
    return found


def test_every_demo_has_a_card_with_a_control_that_starts_on_one_of_its_values(
    laboratory: LaboratoryService,
) -> None:
    cards = laboratory.demos()

    assert [card.name for card in cards] == list(LaboratoryDemo)
    for card in cards:
        assert card.summary
        assert explain(card.topic).text  # o tema do "Porquê?" existe
        assert (card.control is None) == bool(card.variants)  # um controlo, ou um por versão
        # O guia: a ideia, pelo menos três coisas a experimentar, e o que fica.
        assert card.guide.idea
        assert len(card.guide.steps) >= 3
        assert card.guide.takeaway
    regularization = next(card for card in cards if card.name is LaboratoryDemo.REGULARIZATION)
    assert [variant.name for variant in regularization.variants] == ["ridge", "lasso"]
    assert regularization.default_variant == "lasso"
    for control in controls(laboratory).values():
        assert list(control.values) == sorted(set(control.values))
        assert control.default in control.values


def test_every_value_a_control_offers_works(laboratory: LaboratoryService) -> None:
    offered = controls(laboratory)

    # Os extremos de cada controlo (o meio já se usa nos outros testes).
    for degree in (offered["overfitting"].values[0], offered["overfitting"].values[-1]):
        assert laboratory.overfitting(int(degree)).notices
    for rate in (offered["learning_rate"].values[0], offered["learning_rate"].values[-1]):
        assert laboratory.learning_rate(rate).notices
    assert laboratory.knn(int(offered["knn"].values[-1])).boundary is not None
    for kind in ("ridge", "lasso"):
        values = offered[f"regularization.{kind}"].values
        for alpha in (values[0], values[-1]):
            assert laboratory.regularization(kind, alpha).notices
    for threshold in (offered["threshold"].values[0], offered["threshold"].values[-1]):
        assert laboratory.threshold(threshold).notices
    for size in (offered["data_size"].values[0], offered["data_size"].values[-1]):
        assert laboratory.data_size(int(size)).boundary is not None


def test_values_the_demos_cannot_use_are_explained(laboratory: LaboratoryService) -> None:
    with pytest.raises(InvalidParameterError, match="between 1 and 15"):
        laboratory.overfitting(16)
    with pytest.raises(InvalidParameterError, match="positive number"):
        laboratory.learning_rate(0.0)
    with pytest.raises(InvalidParameterError, match="neighbours"):
        laboratory.knn(0)
    with pytest.raises(InvalidParameterError, match="positive number"):
        laboratory.regularization("lasso", 0.0)
    for threshold in (0.0, 1.0, float("nan")):
        with pytest.raises(InvalidParameterError, match="between 0 and 1"):
            laboratory.threshold(threshold)
    for size in (5, 751):
        with pytest.raises(InvalidParameterError, match="between 10 and 750"):
            laboratory.data_size(size)


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


def test_the_threshold_trades_false_alarms_for_missed_cases(
    laboratory: LaboratoryService,
) -> None:
    low, high = laboratory.threshold(0.1), laboratory.threshold(0.9)

    def chosen(demo: ThresholdDemo, values: tuple[float | None, ...]) -> float | None:
        return values[demo.thresholds.index(demo.threshold)]

    # Baixo: encontra quase todos, com muitos alarmes falsos; alto: o contrário.
    assert chosen(low, low.recall) == pytest.approx(0.9, abs=0.05)
    assert (chosen(low, low.precision) or 0) < 0.6
    assert (chosen(high, high.precision) or 0) > 0.9
    assert chosen(high, high.recall) == pytest.approx(0.33, abs=0.1)
    assert [notice.code for notice in low.notices] == ["threshold_low", "accuracy_trap"]
    assert [notice.code for notice in high.notices] == ["threshold_high", "accuracy_trap"]
    # A matriz de confusão: as linhas são a classe real (a soma da segunda são os positivos).
    (tn, fp), (fn, tp) = low.confusion
    assert tn + fp + fn + tp == 1000
    assert (fn + tp) / 1000 == pytest.approx(low.positive_share)
    # Uma probabilidade por linha do teste, de cada classe; os positivos marcados são os que
    # ficam acima do limiar.
    assert (len(low.negatives), len(low.positives)) == (tn + fp, fn + tp)
    assert sum(value >= low.threshold for value in low.positives) == tp
    # A armadilha da exatidão: responder sempre "negativo" já acerta quase 9 em 10.
    assert low.baseline_accuracy == pytest.approx(0.9, abs=0.02)


def test_a_threshold_that_flags_nothing_has_no_precision(laboratory: LaboratoryService) -> None:
    demo = laboratory.threshold(0.9999)

    assert demo.precision[-1] is None
    assert demo.notices[0].code == "flags_nothing"


def test_more_data_helps_the_flexible_model_more_than_the_simple_one(
    laboratory: LaboratoryService,
) -> None:
    few, many = laboratory.data_size(10), laboratory.data_size(750)

    # A árvore sem limite acerta sempre as linhas de treino (decora), mas só generaliza com
    # muitas; a regressão logística aprende depressa e fica-se por aí (o viés).
    assert all(accuracy == pytest.approx(1.0) for accuracy in many.tree_train)
    assert many.tree_test[-1] > many.tree_test[0] + 0.1
    assert many.tree_test[-1] > many.logistic_test[-1]
    assert many.tree_test[0] < many.logistic_test[0]
    hundred = many.sizes.index(100)
    assert many.logistic_test[-1] - many.logistic_test[hundred] < 0.03
    # Com as 750 linhas, as amostras são todas iguais: não há desvio entre elas.
    assert many.tree_spread[-1] == pytest.approx(0.0)
    assert [notice.code for notice in few.notices] == ["simple_wins"]
    assert [notice.code for notice in many.notices] == ["flexible_wins"]
    # A fronteira desenha só as linhas que treinaram a árvore, das duas classes.
    assert few.boundary is not None
    assert len(few.boundary.points.x) == 10
    assert sorted(set(few.boundary.points.labels)) == sorted(few.boundary.class_labels)
