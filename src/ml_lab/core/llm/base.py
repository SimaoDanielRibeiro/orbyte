"""A conversa com um modelo de linguagem: as mensagens, os pedaços da resposta e a interface.

Um modelo de linguagem responde aos bocados (``Chunk``), à medida que gera o texto. Os modelos
que pensam antes de responder (o Qwen3, o DeepSeek-R1) mandam o raciocínio à parte, no campo
``thinking``, e só depois a resposta, em ``content``: é isso que deixa a interface mostrar um
agente a pensar, ao vivo, sem misturar o pensamento com a resposta.

O resto do laboratório só conhece a interface ``LanguageModel``. O Ollama é uma maneira de a
cumprir (``ollama.py``); o ``ScriptedModel`` (``scripted.py``) é outra, com as respostas
escritas à mão, para os testes correrem sem o Ollama (ADR 0042).
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

type Role = Literal["system", "user", "assistant", "tool"]
"""Quem fala numa mensagem: as instruções, o pedido, o modelo ou o resultado de uma ferramenta."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Um pedido do modelo para usar uma ferramenta.

    Attributes:
        name: O nome da ferramenta.
        arguments: Os argumentos, já lidos do JSON.
    """

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    """Uma mensagem da conversa.

    Attributes:
        role: Quem fala: as instruções (``system``), o pedido (``user``), o modelo
            (``assistant``) ou o resultado de uma ferramenta (``tool``).
        content: O texto.
        tool_calls: As ferramentas que o modelo pediu (só nas mensagens do modelo).
        tool_name: A ferramenta que deu este resultado (só nas mensagens ``tool``).
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = None


def _no_arguments() -> dict[str, Any]:
    return {"type": "object", "properties": {}}


@dataclass(frozen=True, slots=True)
class Tool:
    """Uma ferramenta que o modelo pode pedir para usar.

    Attributes:
        name: O nome, como o de uma função (``list_candidates``).
        description: O que faz, em português: é o que o modelo lê para decidir usá-la.
        parameters: Os argumentos, em JSON Schema.
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=_no_arguments)


@dataclass(frozen=True, slots=True)
class Chunk:
    """Um pedaço da resposta, tal como chega.

    Attributes:
        thinking: Um pedaço do raciocínio (vazio nos modelos que não pensam à parte).
        content: Um pedaço da resposta.
        tool_calls: As ferramentas pedidas (chegam inteiras, num pedaço só).
        done: Se é o último pedaço.
    """

    thinking: str = ""
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    done: bool = False


@dataclass(frozen=True, slots=True)
class Reply:
    """A resposta inteira: o raciocínio, o texto e as ferramentas pedidas.

    Attributes:
        thinking: O raciocínio, se o modelo pensou à parte.
        content: A resposta.
        tool_calls: As ferramentas que o modelo pediu.
    """

    thinking: str = ""
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    def as_message(self) -> Message:
        """A resposta como uma mensagem do modelo, para continuar a conversa."""
        return Message("assistant", self.content, self.tool_calls)


class LanguageModel(Protocol):
    """Um modelo de linguagem que responde aos bocados."""

    @property
    def name(self) -> str:
        """O nome do modelo ("qwen3:8b")."""
        ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        think: bool = False,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        seed: int | None = None,
    ) -> Iterator[Chunk]:
        """Responde à conversa, aos bocados.

        Args:
            messages: A conversa até aqui.
            think: Se o modelo pensa antes de responder (o raciocínio vem em ``thinking``).
            schema: Um JSON Schema que a resposta tem de cumprir (a saída estruturada).
            tools: As ferramentas que o modelo pode pedir.
            seed: A seed do sorteio das palavras: com a mesma seed e a mesma conversa, a
                resposta tende a ser a mesma (não é garantido entre computadores diferentes).

        Raises:
            LanguageModelError: Se o modelo não responder.
        """
        ...


def collect(chunks: Iterable[Chunk], on_chunk: Callable[[Chunk], None] | None = None) -> Reply:
    """Junta os pedaços numa resposta inteira.

    Args:
        chunks: Os pedaços, pela ordem em que chegam.
        on_chunk: Chamada a cada pedaço, assim que chega (para mostrar o pensamento ao vivo).
    """
    thinking: list[str] = []
    content: list[str] = []
    calls: list[ToolCall] = []
    for chunk in chunks:
        if on_chunk is not None:
            on_chunk(chunk)
        thinking.append(chunk.thinking)
        content.append(chunk.content)
        calls.extend(chunk.tool_calls)
    return Reply("".join(thinking), "".join(content), tuple(calls))
