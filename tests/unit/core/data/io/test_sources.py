import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.data.io import builtin_source
from ml_lab.core.data.io.base import DATA_SOURCES, SourceSpec, open_source
from ml_lab.core.data.io.builtin_source import SAMPLES, SampleSource
from ml_lab.core.data.io.synthetic_source import GENERATORS, TARGET, SyntheticSource
from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    DataValidationError,
    InvalidParameterError,
)
from ml_lab.core.foundation.types import TaskType

# --- referências às origens ------------------------------------------------------------


def test_there_is_one_source_type_per_origin() -> None:
    assert DATA_SOURCES.names() == ["file", "sample", "synthetic"]


def test_spec_keys_are_stable_and_ignore_parameter_order() -> None:
    first = SourceSpec("synthetic", {"generator": "moons", "seed": 1})
    second = SourceSpec("synthetic", {"seed": 1, "generator": "moons"})

    assert first.key == second.key
    assert first.key != SourceSpec("synthetic", {"generator": "moons", "seed": 2}).key
    assert len(first.key) == 12


def test_opening_unknown_sources_or_parameters_is_explained() -> None:
    with pytest.raises(ComponentNotFoundError, match="Did you mean 'sample'"):
        open_source(SourceSpec("samples", {"name": "iris"}))
    with pytest.raises(InvalidParameterError, match="Invalid parameters"):
        open_source(SourceSpec("sample", {"nome": "iris"}))


# --- datasets de exemplo -----------------------------------------------------------------


@pytest.mark.parametrize("name", SAMPLES.names())
def test_every_sample_loads_with_its_documented_shape(name: str) -> None:
    info = SAMPLES.get(name)

    table = SampleSource(name).load()

    assert table.frame.shape == (info.n_rows, info.n_columns)
    assert table.suggested_target == info.target
    assert info.target in table.frame.columns
    assert table.spec == SourceSpec("sample", {"name": name})


def test_samples_can_be_filtered_by_task() -> None:
    assert sorted(SAMPLES.for_task(TaskType.REGRESSION)) == ["california_housing"]


def test_a_changed_sample_file_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = tmp_path / "data" / "samples"
    shutil.copytree(get_settings().samples_dir, samples)
    iris = samples / "iris.csv"
    iris.write_text(iris.read_text(encoding="utf-8").replace("setosa", "rosa", 1), encoding="utf-8")
    settings = Settings.from_env({}, home=tmp_path)
    monkeypatch.setattr(builtin_source, "get_settings", lambda: settings)

    with pytest.raises(DataValidationError, match="unexpected values: rosa"):
        SampleSource("iris").load()


# --- dados sintéticos ------------------------------------------------------------------


@pytest.mark.parametrize("name", GENERATORS.names())
def test_generators_are_reproducible_with_the_same_seed(name: str) -> None:
    first = SyntheticSource(name, seed=7).load().frame
    second = SyntheticSource(name, seed=7).load().frame
    other = SyntheticSource(name, seed=8).load().frame

    assert first.equals(second)
    assert not first.equals(other)
    assert TARGET in first.columns


def test_moons_without_noise_lie_on_two_half_circles() -> None:
    frame = SyntheticSource("moons", 0, {"n_samples": 100, "noise": 0.0}).load().frame
    upper = frame[frame[TARGET] == 0]
    lower = frame[frame[TARGET] == 1]

    assert np.allclose(upper["x1"] ** 2 + upper["x2"] ** 2, 1.0)
    assert np.allclose((lower["x1"] - 1) ** 2 + (lower["x2"] - 0.5) ** 2, 1.0)
    assert len(upper) == len(lower) == 50


def test_circles_use_the_inner_radius_factor() -> None:
    frame = SyntheticSource("circles", 0, {"n_samples": 60, "noise": 0.0, "factor": 0.3}).load()
    radius = np.hypot(frame.frame["x1"], frame.frame["x2"])

    assert np.allclose(radius[frame.frame[TARGET] == 1], 0.3)
    assert np.allclose(radius[frame.frame[TARGET] == 0], 1.0)


def test_blobs_split_the_points_evenly_between_centers() -> None:
    frame = SyntheticSource("blobs", 3, {"n_samples": 100, "centers": 3}).load().frame

    assert frame[TARGET].value_counts().sort_index().tolist() == [34, 33, 33]


def test_regression_documents_the_rule_it_used() -> None:
    table = SyntheticSource("regression", 5, {"n_samples": 50, "noise": 0.0}).load()
    x = np.column_stack([table.frame["x1"], np.ones(len(table.frame))])
    slope, intercept = np.linalg.lstsq(x, table.frame[TARGET], rcond=None)[0]

    # Sem ruído, a reta reencontra exatamente a regra (escrita com ponto decimal).
    assert f"{slope:.1f}" in table.description
    assert f"{intercept:.1f}" in table.description


def test_the_spec_stores_every_parameter_including_defaults() -> None:
    spec = SyntheticSource("moons", 1).spec()

    assert spec.params == {
        "generator": "moons",
        "seed": 1,
        "parameters": {"n_samples": 200, "noise": 0.2},
    }


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"n_points": 10}, "has no parameters n_points"),
        ({"n_samples": 10}, "between 20 and 5,000"),
        ({"n_samples": 50.5}, "whole number"),
        ({"noise": "muito"}, "must be a number"),
    ],
)
def test_generator_parameters_are_validated(parameters: dict[str, object], message: str) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        GENERATORS.get("moons").resolve(parameters)


def test_parameters_are_described_in_json_schema() -> None:
    schema = GENERATORS.get("circles").parameters_schema()

    assert schema["required"] == ["n_samples", "noise", "factor"]
    assert schema["properties"]["n_samples"] == {
        "type": "integer",
        "title": "Number of points",
        "description": "How many rows to generate.",
        "default": 200,
        "minimum": 20,
        "maximum": 5000,
    }
    assert math.isclose(schema["properties"]["factor"]["default"], 0.5)
