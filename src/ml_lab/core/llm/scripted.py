"""Um modelo de linguagem com as respostas escritas à mão: os testes correm sem o Ollama.

Cada chamada a ``chat`` dá a resposta seguinte da lista, aos bocados, como um modelo a sério:
primeiro o raciocínio (se se pediu ``think``), uma palavra de cada vez, depois a resposta e,
no fim, as ferramentas pedidas. Guarda o que recebeu em cada chamada, para os testes verem o
que o agente perguntou ao modelo.
"""

import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_lab.core.foundation.errors import LanguageModelError
from ml_lab.core.llm.base import Chunk, Message, Reply, Tool


@dataclass(frozen=True, slots=True)
class ScriptedCall:
    """O que o modelo escrito à mão recebeu numa chamada.

    Attributes:
        messages: A conversa.
        think: Se se pediu o raciocínio.
        schema: O JSON Schema pedido para a resposta.
        tools: As ferramentas oferecidas.
        seed: A seed pedida.
    """

    messages: tuple[Message, ...]
    think: bool
    schema: Mapping[str, Any] | None
    tools: tuple[Tool, ...]
    seed: int | None


class ScriptedModel:
    """Um ``LanguageModel`` que responde o que lhe escreveram, pela ordem.

    Args:
        replies: As respostas, uma por chamada.
        name: O nome a mostrar.
    """

    def __init__(self, replies: Iterable[Reply], *, name: str = "hand-written") -> None:
        self._replies = deque(replies)
        self._name = name
        self.calls: list[ScriptedCall] = []

    @property
    def name(self) -> str:
        """O nome do modelo."""
        return self._name

    def chat(
        self,
        messages: Sequence[Message],
        *,
        think: bool = False,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        seed: int | None = None,
    ) -> Iterator[Chunk]:
        """Dá a resposta seguinte, aos bocados (ver ``LanguageModel.chat``).

        Raises:
            LanguageModelError: Se já não houver respostas (o teste pediu mais do que previu).
        """
        self.calls.append(ScriptedCall(tuple(messages), think, schema, tuple(tools), seed))
        if not self._replies:
            raise LanguageModelError(
                "The hand-written model has no replies left: the test asked for more than it "
                "planned."
            )
        return _pieces(self._replies.popleft(), think=think)


def _pieces(reply: Reply, *, think: bool) -> Iterator[Chunk]:
    """A resposta aos bocados: uma palavra de cada vez (com o espaço que vem a seguir)."""
    if think:
        for word in _words(reply.thinking):
            yield Chunk(thinking=word)
    for word in _words(reply.content):
        yield Chunk(content=word)
    yield Chunk(tool_calls=reply.tool_calls, done=True)


def _words(text: str) -> list[str]:
    # Corta-se depois de cada espaço: juntar os pedaços dá o texto igual, espaços incluídos.
    return [piece for piece in re.split(r"(?<=\s)", text) if piece]
