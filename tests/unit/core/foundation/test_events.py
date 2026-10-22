from dataclasses import FrozenInstanceError
from datetime import UTC

import pytest

import ml_lab.core.foundation.events as events_module
from ml_lab.core.foundation.events import (
    Event,
    EventBus,
    RunFailed,
    RunFinished,
    RunStarted,
    SnapshotTaken,
    StepCompleted,
    Subscription,
)


def step(number: int, loss: float = 0.1) -> StepCompleted:
    return StepCompleted(run_id="run-1", step=number, loss=loss)


# --- subscrever e publicar -----------------------------------------------------------


def test_subscriber_receives_published_events() -> None:
    bus = EventBus()
    received: list[StepCompleted] = []
    bus.subscribe(StepCompleted, received.append)

    bus.publish(step(1))
    bus.publish(step(2))

    assert [event.step for event in received] == [1, 2]


def test_subscribers_only_receive_their_event_type() -> None:
    bus = EventBus()
    steps: list[StepCompleted] = []
    snapshots: list[SnapshotTaken] = []
    bus.subscribe(StepCompleted, steps.append)
    bus.subscribe(SnapshotTaken, snapshots.append)

    bus.publish(step(1))
    bus.publish(SnapshotTaken(run_id="run-1", step=1, state={"weights": [0.5, -0.2]}))

    assert len(steps) == 1
    assert len(snapshots) == 1
    assert snapshots[0].state == {"weights": [0.5, -0.2]}


def test_subscribing_to_the_base_event_receives_everything() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(Event, received.append)

    bus.publish(RunStarted(run_id="run-1", n_samples=150, n_features=4))
    bus.publish(step(1))
    bus.publish(RunFinished(run_id="run-1", duration_seconds=0.5))

    assert [event.kind for event in received] == ["run_started", "step", "run_finished"]


def test_subscribers_are_called_in_subscription_order() -> None:
    bus = EventBus()
    calls: list[str] = []
    bus.subscribe(StepCompleted, lambda _: calls.append("gravador"))
    bus.subscribe(StepCompleted, lambda _: calls.append("interface"))

    bus.publish(step(1))

    assert calls == ["gravador", "interface"]


def test_publishing_without_subscribers_does_nothing() -> None:
    EventBus().publish(step(1))


# --- cancelar subscrição -------------------------------------------------------------


def test_unsubscribed_handler_stops_receiving_events() -> None:
    bus = EventBus()
    received: list[StepCompleted] = []
    subscription = bus.subscribe(StepCompleted, received.append)

    bus.publish(step(1))
    bus.unsubscribe(subscription)
    bus.publish(step(2))

    assert [event.step for event in received] == [1]
    assert bus.subscription_count == 0


def test_unsubscribing_twice_is_harmless() -> None:
    bus = EventBus()
    subscription = bus.subscribe(StepCompleted, lambda _: None)

    bus.unsubscribe(subscription)
    bus.unsubscribe(subscription)

    assert bus.subscription_count == 0


def test_only_the_cancelled_subscription_is_removed() -> None:
    bus = EventBus()
    received: list[StepCompleted] = []
    first = bus.subscribe(StepCompleted, received.append)
    bus.subscribe(StepCompleted, received.append)

    bus.unsubscribe(first)
    bus.publish(step(1))

    assert len(received) == 1
    assert bus.subscription_count == 1


def test_handler_can_unsubscribe_itself_while_handling_an_event() -> None:
    bus = EventBus()
    received: list[StepCompleted] = []
    subscriptions: list[Subscription[StepCompleted]] = []

    def handle_once(event: StepCompleted) -> None:
        received.append(event)
        bus.unsubscribe(subscriptions[0])

    subscriptions.append(bus.subscribe(StepCompleted, handle_once))
    bus.publish(step(1))
    bus.publish(step(2))

    assert [event.step for event in received] == [1]


def test_handler_cancelled_during_a_publication_is_not_called() -> None:
    bus = EventBus()
    calls: list[str] = []
    subscriptions: list[Subscription[StepCompleted]] = []

    def cancel_the_next(_: StepCompleted) -> None:
        calls.append("primeiro")
        bus.unsubscribe(subscriptions[1])

    subscriptions.append(bus.subscribe(StepCompleted, cancel_the_next))
    subscriptions.append(bus.subscribe(StepCompleted, lambda _: calls.append("segundo")))
    bus.publish(step(1))

    assert calls == ["primeiro"]


# --- erros e eventos -----------------------------------------------------------------


def test_handler_errors_reach_the_publisher() -> None:
    bus = EventBus()

    def broken(_: StepCompleted) -> None:
        raise RuntimeError("o subscritor falhou")

    bus.subscribe(StepCompleted, broken)

    with pytest.raises(RuntimeError, match="o subscritor falhou"):
        bus.publish(step(1))


def test_events_are_immutable() -> None:
    event = step(1)

    with pytest.raises(FrozenInstanceError):
        event.step = 2  # type: ignore[misc]


def test_events_get_a_utc_timestamp() -> None:
    assert step(1).timestamp.tzinfo is UTC


def test_every_event_type_has_a_unique_kind() -> None:
    # @dataclass(slots=True) cria uma classe nova: a original continua em __subclasses__()
    # até o garbage collector a apagar. Por isso contam-se só as classes do módulo.
    event_types = [
        event_type
        for event_type in Event.__subclasses__()
        if getattr(events_module, event_type.__name__, None) is event_type
    ]
    kinds = [event_type.kind for event_type in event_types]

    assert len(event_types) == 6
    assert len(kinds) == len(set(kinds))
    assert "event" not in kinds


def test_failure_event_carries_an_explanation() -> None:
    event = RunFailed(run_id="run-1", message="O dataset não tem coluna-alvo.", error_type="E")

    assert event.kind == "run_failed"
    assert event.message == "O dataset não tem coluna-alvo."
