"""Logs estruturados do ML Lab, só com o módulo ``logging`` da biblioteca padrão.

Cada módulo cria o seu logger e escreve mensagens com campos extra, que ficam nos
logs como dados estruturados::

    logger = get_logger(__name__)
    logger.info("treino terminado", extra={"run_id": "run-1", "loss": 0.25})

No formato ``text`` (para ler no terminal) a linha fica::

    2026-10-22 14:03:12 INFO ml_lab.core.training: treino terminado | run_id=run-1 loss=0.25

No formato ``json`` (para ferramentas) sai um objeto JSON por linha, com os
mesmos campos.

Os módulos da fundação usam diretamente ``logging.getLogger(__name__)``: como o
nome começa por ``ml_lab``, herdam esta configuração sem importar este módulo.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final, Literal, TextIO, override

type LogFormat = Literal["text", "json"]

LOGGER_NAME: Final = "ml_lab"
LOG_LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMATS: Final[tuple[LogFormat, ...]] = ("text", "json")

_HANDLER_NAME: Final = "ml_lab.console"
# Atributos que todos os LogRecord têm; os restantes vieram do argumento extra=.
_STANDARD_ATTRIBUTES: Final = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def get_logger(name: str) -> logging.Logger:
    """Devolve o logger de um módulo, pendurado no logger ``ml_lab``.

    Args:
        name: Normalmente ``__name__``. Nomes fora de ``ml_lab`` (por exemplo os
            módulos da interface) passam a ``ml_lab.<name>`` para herdarem a
            configuração.
    """
    if name != LOGGER_NAME and not name.startswith(f"{LOGGER_NAME}."):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: str = "INFO", fmt: LogFormat = "text", *, stream: TextIO | None = None
) -> logging.Logger:
    """Configura os logs do ML Lab.

    Pode chamar-se várias vezes (o Streamlit, por exemplo, volta a correr o
    script a cada interação): cada chamada substitui a configuração anterior.
    Só mexe no logger ``ml_lab``; as bibliotecas externas mantêm a sua.

    Args:
        level: Nível mínimo: DEBUG, INFO, WARNING, ERROR ou CRITICAL.
        fmt: ``"text"`` para ler no terminal ou ``"json"`` para logs estruturados.
        stream: Para onde escrever; por omissão, a saída de erro (``sys.stderr``).

    Returns:
        O logger ``ml_lab`` já configurado.

    Raises:
        ValueError: Se o nível ou o formato não forem válidos.
    """
    normalized_level = level.strip().upper()
    if normalized_level not in LOG_LEVELS:
        raise ValueError(
            f"Nível de log inválido '{level}'. Usa um destes: {', '.join(LOG_LEVELS)}."
        )
    try:
        formatter = _FORMATTERS[fmt]()
    except KeyError:
        raise ValueError(
            f"Formato de log inválido '{fmt}'. Usa um destes: {', '.join(LOG_FORMATS)}."
        ) from None

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.name = _HANDLER_NAME
    handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAME)
    for previous in [h for h in logger.handlers if h.name == _HANDLER_NAME]:
        logger.removeHandler(previous)
        previous.close()
    logger.addHandler(handler)
    logger.setLevel(normalized_level)
    return logger


class TextFormatter(logging.Formatter):
    """Formato para ler no terminal: ``data nível logger: mensagem | campo=valor``."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    @override
    def formatMessage(self, record: logging.LogRecord) -> str:
        line = super().formatMessage(record)
        extra = _extra_fields(record)
        if not extra:
            return line
        return f"{line} | " + " ".join(f"{key}={value}" for key, value in extra.items())


class JsonFormatter(logging.Formatter):
    """Logs estruturados: um objeto JSON por linha, com os campos extra incluídos."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in _extra_fields(record).items():
            payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_FORMATTERS: Final[dict[str, type[logging.Formatter]]] = {
    "text": TextFormatter,
    "json": JsonFormatter,
}


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_ATTRIBUTES and not key.startswith("_")
    }
