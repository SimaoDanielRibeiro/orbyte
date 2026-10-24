"""A sequência das camadas de uma rede ao longo do treino, a partir dos snapshots."""

from typing import Any

import pytest

from ml_lab.core.visualization_data.network import network_trace


def layers(gradient: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "Camada 1",
            "units": 2,
            "activation": "ReLU",
            "parameters": 6,
            "gradient": gradient,
            "activations": [0.5, 0.0],
            "stuck_units": [False, True],
            "stuck": 0.5,
            "histogram": {"edges": [0.0, 0.5, 1.0], "counts": [3, 1]},
        },
        {"name": "Saída", "units": 2, "activation": "softmax", "parameters": 6, "gradient": 0.1},
    ]


def test_the_trace_follows_the_snapshots_in_order() -> None:
    snapshots = [(5, {"layers": layers(0.2)}), (1, {"layers": layers(0.9)})]

    trace = network_trace(2, snapshots)

    assert trace is not None
    assert [frame.step for frame in trace.frames] == [1, 5]
    assert [layer.name for layer in trace.layers] == ["Camada 1", "Saída"]
    assert [layer.hidden for layer in trace.layers] == [True, False]
    first = trace.frames[0].layers[0]
    assert (first.gradient, first.stuck, first.stuck_units) == (0.9, 0.5, (False, True))
    assert first.counts == (3, 1)
    assert trace.frames[0].layers[1].activations == ()
    assert trace.frames[0].stuck == 0.5


def test_the_share_of_stuck_units_counts_every_hidden_neuron() -> None:
    hidden, output = layers(0.5)
    wide = hidden | {
        "name": "Camada 2",
        "units": 4,
        "activations": [0.1] * 4,
        "stuck_units": [False] * 4,
        "stuck": 0.0,
    }

    trace = network_trace(2, [(1, {"layers": [hidden, wide, output]})])

    # Um parado em 6 neurónios, e não a média das duas camadas (25%).
    assert trace is not None
    assert trace.frames[0].stuck == pytest.approx(1 / 6)


def test_a_repeated_step_keeps_the_last_snapshot() -> None:
    # O Trainer junta a fotografia final, que pode repetir o passo da última do modelo.
    trace = network_trace(2, [(3, {"layers": layers(0.5)}), (3, {"layers": layers(0.7)})])

    assert trace is not None
    assert len(trace.frames) == 1
    assert trace.frames[0].layers[0].gradient == 0.7


def test_a_model_that_is_not_a_network_has_no_trace() -> None:
    assert network_trace(2, [(1, {"coefficients": [1.0]}), (2, {})]) is None
    assert network_trace(2, []) is None
