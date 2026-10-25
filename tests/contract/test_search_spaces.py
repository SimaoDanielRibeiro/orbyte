"""Contrato dos espaços de procura: o que cada modelo diz aos agentes para experimentar é válido.

Cada modelo declara, no ``x-search`` dos seus hiperparâmetros, o que os agentes da fase 9
podem sortear (ADR 0025). Um modelo novo fica testado aqui sem escrever testes novos: todos
os valores da lista, os extremos de cada intervalo e cem sorteios têm de passar a validação
do próprio modelo.
"""

from typing import Any

import numpy as np
import pytest

from ml_lab.core.automl.objective import OBJECTIVES
from ml_lab.core.automl.search_space import search_space
from ml_lab.core.models.base import MODELS, LabModel

ALL = [model for _, model in MODELS.items()]
SUPERVISED = [model for model in ALL if model.info.task_types & set(OBJECTIVES)]

NOTHING_TO_TUNE = {
    "linear_regression": "só tem fit_intercept, que não muda a capacidade do modelo",
    "linear_regression_gd": "chega aos mesmos pesos que a de mínimos quadrados",
    "perceptron": "didático; a taxa de aprendizagem não muda a fronteira a que chega",
}
"""Os modelos supervisionados sem espaço de procura, e porquê."""


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.info.name)
def test_the_search_space_of_every_model_reads(model: type[LabModel[Any]]) -> None:
    space = search_space(model)

    assert space.model == model.info.name
    fields = model.info.hyperparameters.model_fields
    assert all(dimension.name in fields for dimension in space.dimensions)


@pytest.mark.parametrize("model", SUPERVISED, ids=lambda model: model.info.name)
def test_every_value_the_agents_can_try_is_valid(model: type[LabModel[Any]]) -> None:
    space = search_space(model)
    parse = model.info.hyperparameters.parse
    for dimension in space.dimensions:
        if dimension.values:
            for value in dimension.values:
                parse({dimension.name: value}, model=model.info.name)
        else:
            low, high = dimension.bounds
            ends = (int(low), int(high)) if dimension.integer else (low, high)
            for value in ends:
                parse({dimension.name: value}, model=model.info.name)
    rng = np.random.default_rng(0)
    for _ in range(100):
        parse(space.sample(rng), model=model.info.name)


def test_every_supervised_model_says_what_to_search_or_why_not() -> None:
    untuned = {model.info.name for model in SUPERVISED if search_space(model).is_empty}

    assert untuned == set(NOTHING_TO_TUNE), (
        "Declara o x-search dos hiperparâmetros que valem a pena procurar (ver "
        "schemas/hyperparameters.py), ou junta o modelo a NOTHING_TO_TUNE com a razão."
    )
