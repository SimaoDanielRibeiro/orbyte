"""Avisos didáticos: o que merece atenção, explicado em vez de escondido.

Cada etapa do ML Lab pode produzir avisos ("30% de nulos na coluna Cabin", "a
classe positiva só tem 5% das linhas"). Um aviso tem um nível, um código estável
(para testes e para a interface os agrupar) e uma mensagem em português.

Exemplo:
    >>> notice = Notice.warning("missing_values", "A coluna Age tem 19,9% de nulos.", "Age")
    >>> notice.level is NoticeLevel.WARNING
    True
    >>> notice.subject
    'Age'
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class NoticeLevel(StrEnum):
    """Gravidade de um aviso.

    - ``INFO``: uma nota útil, sem nada para corrigir;
    - ``WARNING``: algo que pode prejudicar o resultado e vale a pena rever;
    - ``ERROR``: algo que impede continuar.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Notice:
    """Um aviso didático.

    Attributes:
        level: Gravidade do aviso.
        code: Código estável em ``snake_case``, por exemplo ``"missing_values"``.
        message: Explicação em português, com o que fazer quando há alguma coisa a fazer.
        subject: Aquilo a que o aviso se refere (uma coluna, uma métrica...), se houver.
    """

    level: NoticeLevel
    code: str
    message: str
    subject: str | None = None

    @classmethod
    def info(cls, code: str, message: str, subject: str | None = None) -> Self:
        """Cria um aviso de nível ``INFO``."""
        return cls(NoticeLevel.INFO, code, message, subject)

    @classmethod
    def warning(cls, code: str, message: str, subject: str | None = None) -> Self:
        """Cria um aviso de nível ``WARNING``."""
        return cls(NoticeLevel.WARNING, code, message, subject)

    @classmethod
    def error(cls, code: str, message: str, subject: str | None = None) -> Self:
        """Cria um aviso de nível ``ERROR``."""
        return cls(NoticeLevel.ERROR, code, message, subject)


_ORDER = {NoticeLevel.ERROR: 0, NoticeLevel.WARNING: 1, NoticeLevel.INFO: 2}


def sort_notices(notices: Iterable[Notice]) -> tuple[Notice, ...]:
    """Ordena os avisos do mais grave para o menos grave, mantendo a ordem dentro de cada nível."""
    return tuple(sorted(notices, key=lambda notice: _ORDER[notice.level]))


def has_errors(notices: Iterable[Notice]) -> bool:
    """Se algum dos avisos é um erro, isto é, se impede continuar."""
    return any(notice.level is NoticeLevel.ERROR for notice in notices)
