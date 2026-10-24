"""Serialização da fronteira: resultados em JSON, esquemas e a marca dos métodos internos."""

import json
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import PydanticSchemaGenerationError

from ml_lab.core.foundation.types import Partition
from ml_lab.core.schemas.serialization import internal, is_internal, json_schema, to_json


@dataclass(frozen=True, slots=True)
class Point:
    loss: float
    partition: Partition
    tags: tuple[str, ...] = ()


def test_results_become_plain_json_values() -> None:
    data = to_json({Partition.TEST: Point(0.5, Partition.TRAIN, ("a",))})

    assert data == {"test": {"loss": 0.5, "partition": "train", "tags": ["a"]}}


def test_numbers_json_does_not_have_become_null() -> None:
    # Uma loss que explodiu (infinito) ou indefinida (NaN) não pode chegar ao navegador.
    data = to_json(Point(float("inf"), Partition.TEST))

    assert data["loss"] is None
    json.dumps(data, allow_nan=False)


def test_numpy_values_in_free_fields_become_plain_numbers() -> None:
    state = {"weights": np.array([1.5, np.nan]), "iterations": np.int64(3)}

    assert to_json(state) == {"weights": [1.5, None], "iterations": 3}


def test_the_schema_describes_a_result_and_refuses_what_json_cannot_hold() -> None:
    schema = json_schema(Point)

    assert set(schema["properties"]) == {"loss", "partition", "tags"}
    with pytest.raises(PydanticSchemaGenerationError):
        json_schema(NDArray[np.float64])


def test_internal_methods_carry_their_mark() -> None:
    @internal
    def for_other_services() -> None:
        """Só para outros serviços."""

    assert is_internal(for_other_services)
    assert not is_internal(to_json)
