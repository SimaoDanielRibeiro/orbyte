import pytest

from ml_lab.core.foundation.errors import LanguageModelError
from ml_lab.core.llm import Message, Reply, ScriptedModel, ToolCall, collect


def test_it_answers_in_pieces_that_rebuild_the_reply() -> None:
    reply = Reply("Primeiro penso.  Depois\nrespondo.", '{"k": 5}', (ToolCall("look", {}),))
    model = ScriptedModel([reply])

    chunks = list(model.chat([Message("user", "Olá")], think=True))

    assert len(chunks) > 3  # aos bocados, como um modelo a sério
    assert chunks[0].thinking == "Primeiro "
    assert chunks[-1].done
    assert collect(chunks) == reply


def test_without_think_the_reasoning_does_not_arrive() -> None:
    model = ScriptedModel([Reply("Isto não se vê.", "Resposta.")])

    reply = collect(model.chat([Message("user", "Olá")]))

    assert reply == Reply("", "Resposta.")


def test_it_remembers_what_it_was_asked() -> None:
    model = ScriptedModel([Reply(content="1"), Reply(content="2")])
    schema = {"type": "object"}

    collect(model.chat([Message("user", "a")], think=True, schema=schema, seed=3))
    collect(model.chat([Message("user", "b")]))

    assert [call.messages[0].content for call in model.calls] == ["a", "b"]
    assert (model.calls[0].think, model.calls[0].schema, model.calls[0].seed) == (True, schema, 3)


def test_asking_more_than_was_written_is_an_error() -> None:
    model = ScriptedModel([])

    with pytest.raises(LanguageModelError, match="has no replies left"):
        model.chat([Message("user", "Olá")])
