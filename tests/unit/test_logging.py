import io
import json
import logging
from collections.abc import Iterator

import pytest

from ml_lab.logging import LOGGER_NAME, configure_logging, get_logger


@pytest.fixture(autouse=True)
def restore_ml_lab_logger() -> Iterator[None]:
    """A configuração dos logs é global: cada teste deixa-a como a encontrou."""
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level = list(logger.handlers), logger.level
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_text_format_appends_the_extra_fields() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "text", stream=stream)

    get_logger("ml_lab.tests").info("treino terminado", extra={"run_id": "run-1", "loss": 0.25})

    assert "INFO ml_lab.tests: treino terminado | run_id=run-1 loss=0.25" in stream.getvalue()


def test_json_format_writes_one_object_per_line() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream=stream)
    logger = get_logger("ml_lab.tests")

    logger.warning("classe positiva com %d%% das linhas", 5, extra={"column": "sobreviveu"})
    logger.info("segunda linha")

    first, second = (json.loads(line) for line in stream.getvalue().splitlines())
    assert first["level"] == "WARNING"
    assert first["logger"] == "ml_lab.tests"
    assert first["message"] == "classe positiva com 5% das linhas"
    assert first["column"] == "sobreviveu"
    assert "timestamp" in first
    assert second["message"] == "segunda linha"


def test_json_format_includes_the_exception() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream=stream)

    try:
        raise ValueError("coluna em falta")
    except ValueError:
        get_logger("ml_lab.tests").exception("a leitura falhou")

    assert "ValueError: coluna em falta" in json.loads(stream.getvalue())["exception"]


def test_messages_below_the_level_are_filtered() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)

    get_logger("ml_lab.tests").info("não aparece")

    assert stream.getvalue() == ""


def test_configuring_twice_replaces_the_previous_configuration() -> None:
    first, second = io.StringIO(), io.StringIO()
    configure_logging("INFO", stream=first)
    configure_logging("INFO", stream=second)

    get_logger("ml_lab.tests").info("uma vez")

    assert first.getvalue() == ""
    assert second.getvalue().count("uma vez") == 1


@pytest.mark.parametrize(("level", "fmt"), [("VERBOSE", "text"), ("INFO", "xml")])
def test_invalid_level_or_format_is_rejected(level: str, fmt: str) -> None:
    with pytest.raises(ValueError, match="inválido"):
        configure_logging(level, fmt, stream=io.StringIO())  # type: ignore[arg-type]


def test_get_logger_hangs_every_logger_under_ml_lab() -> None:
    assert get_logger("ml_lab.core.training").name == "ml_lab.core.training"
    assert get_logger("ml_lab").name == "ml_lab"
    assert get_logger("components.layout").name == "ml_lab.components.layout"


def test_root_logger_is_left_alone() -> None:
    root_handlers = list(logging.getLogger().handlers)

    configure_logging("DEBUG", stream=io.StringIO())

    assert logging.getLogger().handlers == root_handlers
