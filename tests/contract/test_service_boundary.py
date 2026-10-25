"""Contrato da fronteira dos serviços: tudo o que a interface (e a API) recebe passa a JSON.

A interface aprimorada (fase 10) vai falar com o núcleo através de uma API HTTP (fase 7).
Para isso, cada método público dos serviços devolve dados simples, que o Pydantic sabe
descrever (o JSON Schema, de onde o FastAPI tira o OpenAPI) e passar a JSON. Um método
novo fica testado aqui sem escrever testes novos. Os métodos marcados com ``@internal``
servem só outros serviços (devolvem DataFrames ou o modelo treinado) e ficam de fora
(ADR 0019).
"""

import inspect
import json
from collections.abc import Callable
from typing import Any, get_origin, get_type_hints

import pytest

from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import ModelFamily
from ml_lab.core.schemas.automl import AutoMLRequest, SearchSettings
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.schemas.serialization import from_json, is_internal, json_schema, to_json
from ml_lab.core.services import vocabulary
from ml_lab.core.services.lab import Lab
from ml_lab.core.training.jobs import TrainingJob


def public_methods() -> list[tuple[str, Callable[..., Any]]]:
    """Os métodos públicos de cada serviço do ``Lab``, sem os marcados como internos."""
    found: list[tuple[str, Callable[..., Any]]] = []
    for service, service_class in get_type_hints(Lab).items():
        for name, method in inspect.getmembers(service_class, inspect.isfunction):
            if not name.startswith("_") and not is_internal(method):
                found.append((f"{service}.{name}", method))
    return found


PUBLIC = public_methods()


def test_the_discovery_finds_the_public_methods() -> None:
    names = [name for name, _ in PUBLIC]

    assert "experiments.run" in names
    assert "datasets.prepared" not in names  # marcado como interno


@pytest.mark.parametrize("method", [method for _, method in PUBLIC], ids=[n for n, _ in PUBLIC])
def test_every_public_method_returns_a_type_json_can_describe(method: Callable[..., Any]) -> None:
    returned = get_type_hints(method)["return"]
    if get_origin(returned) is TrainingJob:
        # O treino vivo fica do lado do servidor: o que sai para fora é o seu estado.
        returned = get_type_hints(TrainingJob.state)["return"]

    assert json_schema(returned)


@pytest.fixture(scope="module")
def lab() -> Lab:
    return Lab.create()


def test_real_results_become_strict_json(lab: Lab) -> None:
    iris = lab.datasets.load_sample("iris")
    supervised = PreparationConfig(target="species")
    config = ExperimentConfig(
        dataset=iris.source,
        preparation=supervised,
        model=ModelSpec(name="logistic_regression_gd"),
        seed=0,
    )
    job = lab.experiments.start(config)
    result = job.wait(timeout=60)
    clustering = lab.experiments.run(
        config.model_copy(
            update={
                "preparation": PreparationConfig(target="species", task_type=TaskType.CLUSTERING),
                "model": ModelSpec(name="kmeans"),
            }
        )
    )
    search = AutoMLRequest(
        dataset=iris.source,
        target="species",
        families=(ModelFamily.LINEAR, ModelFamily.TREE),
        settings=SearchSettings(budget=3, folds=3),
        seed=0,
    )
    automl = lab.automl.start(search)
    automl.wait(timeout=60)
    outputs = [
        lab.automl.options(TaskType.BINARY_CLASSIFICATION),
        lab.automl.plan(search),
        lab.automl.updates(automl.run_id),
        lab.automl.result(automl.run_id),
        lab.datasets.profile(iris.id, "species"),
        lab.datasets.prepare(iris.id, supervised, 0),
        lab.catalog.models(),
        job.state(),
        result,
        clustering,
        lab.analysis.charts(result.run_id),
        lab.analysis.charts(clustering.run_id),
        lab.analysis.cross_validation(result.run_id),
        lab.analysis.permutation_importance(result.run_id),
        lab.analysis.partial_dependence(result.run_id, result.feature_names[0]),
        lab.analysis.explain_row(result.run_id, 0),
        lab.laboratory.knn(5),
        vocabulary(),
    ]

    for output in outputs:
        data = to_json(output)
        json.dumps(data, allow_nan=False)
        # E o JSON cumpre o próprio esquema: um NaN que passou a null num campo que o esquema
        # diz ser um número mentia aos tipos TypeScript do frontend.
        items, documents = (output, data) if isinstance(output, list) else ([output], [data])
        for item, document in zip(items, documents, strict=True):
            from_json(type(item), document)
