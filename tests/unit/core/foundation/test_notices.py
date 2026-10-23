from dataclasses import FrozenInstanceError

import pytest

from ml_lab.core.foundation.notices import Notice, NoticeLevel, has_errors, sort_notices


def test_constructors_set_the_level() -> None:
    assert Notice.info("a", "nota").level is NoticeLevel.INFO
    assert Notice.warning("b", "aviso").level is NoticeLevel.WARNING
    assert Notice.error("c", "erro", "Age").subject == "Age"


def test_notices_are_sorted_from_most_to_least_severe_keeping_their_order() -> None:
    notices = [
        Notice.info("i1", "nota 1"),
        Notice.warning("w1", "aviso 1"),
        Notice.error("e1", "erro"),
        Notice.warning("w2", "aviso 2"),
    ]

    assert [notice.code for notice in sort_notices(notices)] == ["e1", "w1", "w2", "i1"]


def test_only_errors_block() -> None:
    assert not has_errors([Notice.warning("w", "aviso"), Notice.info("i", "nota")])
    assert has_errors([Notice.info("i", "nota"), Notice.error("e", "erro")])


def test_notices_are_immutable() -> None:
    notice = Notice.info("a", "nota")

    with pytest.raises(FrozenInstanceError):
        notice.message = "outra"  # type: ignore[misc]
