"""O Ollama: um servidor que corre modelos de linguagem no próprio computador.

Fala-se com ele em HTTP, com a biblioteca padrão (``urllib``), sem dependências novas:

- ``GET /api/version`` e ``GET /api/tags``: se está a correr, e que modelos tem;
- ``POST /api/show``: o que um modelo sabe fazer (``"thinking"`` quer dizer que pensa à parte,
  antes de responder, e o agente de IA precisa disso);
- ``POST /api/chat``: a conversa. Com ``"stream": true`` a resposta chega aos bocados, um
  objeto JSON por linha (NDJSON); com ``"think": true`` o raciocínio vem à parte, em
  ``message.thinking``; com ``"format"`` (um JSON Schema) a resposta tem de o cumprir; com
  ``"tools"`` o modelo pode pedir para usar uma ferramenta (``message.tool_calls``).

A temperatura vai sempre a 0: o modelo escolhe a palavra mais provável em vez de sortear, e a
mesma pergunta tende a dar a mesma resposta (ADR 0042).

Só este módulo sabe que o Ollama existe (há uma regra de arquitetura): o resto do laboratório
usa a interface ``LanguageModel``.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ml_lab.core.foundation.errors import LanguageModelError
from ml_lab.core.llm.base import Chunk, Message, Tool, ToolCall

TEMPERATURE: Final = 0.0
"""A temperatura do sorteio das palavras: 0 escolhe sempre a mais provável."""

STATUS_TIMEOUT: Final = 2.0
"""Quanto se espera pelo Ollama para saber se está a correr (o ecrã não pode ficar parado)."""

CHAT_TIMEOUT: Final = 300.0
"""Quanto se espera por cada pedaço da resposta. Na primeira conversa, o Ollama carrega o
modelo para a gráfica, e isso pode demorar."""


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """O que o Ollama diz de si.

    Attributes:
        version: A versão do Ollama.
        models: Os modelos descarregados ("qwen3:8b"), por ordem alfabética.
    """

    version: str
    models: tuple[str, ...]


class OllamaServer:
    """O servidor do Ollama, num endereço.

    Args:
        url: O endereço ("http://localhost:11434"), sem a barra no fim.
        timeout: Quanto se espera por cada resposta, em segundos.
    """

    def __init__(self, url: str, *, timeout: float = STATUS_TIMEOUT) -> None:
        self.url = url
        self._timeout = timeout

    def info(self) -> ServerInfo:
        """A versão e os modelos do Ollama.

        Raises:
            LanguageModelError: Se o Ollama não responder.
        """
        version = _get_json(f"{self.url}/api/version", self._timeout).get("version", "")
        tags = _get_json(f"{self.url}/api/tags", self._timeout)
        models = sorted(str(model.get("name", "")) for model in tags.get("models") or ())
        return ServerInfo(str(version), tuple(name for name in models if name))

    def capabilities(self, name: str) -> tuple[str, ...] | None:
        """O que um modelo sabe fazer, segundo o Ollama ("completion", "thinking", "tools"…).

        Returns:
            As capacidades, ou ``None`` se este Ollama não as disser (as versões antigas não
            o faziam).

        Raises:
            LanguageModelError: Se o Ollama não responder.
        """
        found = _post_json(f"{self.url}/api/show", {"model": name}, self._timeout).get(
            "capabilities"
        )
        if not isinstance(found, list):
            return None
        return tuple(str(item) for item in found)

    def model(self, name: str, *, timeout: float = CHAT_TIMEOUT) -> "OllamaModel":
        """Um modelo deste servidor, pronto para conversar."""
        return OllamaModel(self.url, name, timeout=timeout)


class OllamaModel:
    """Um modelo do Ollama: cumpre a interface ``LanguageModel``.

    Args:
        url: O endereço do Ollama.
        name: O nome do modelo ("qwen3:8b").
        timeout: Quanto se espera por cada pedaço da resposta, em segundos.
    """

    def __init__(self, url: str, name: str, *, timeout: float = CHAT_TIMEOUT) -> None:
        self.url = url
        self._name = name
        self._timeout = timeout

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
        """Responde à conversa, aos bocados (ver ``LanguageModel.chat``)."""
        body = chat_body(self._name, messages, think=think, schema=schema, tools=tools, seed=seed)
        request = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=self._timeout)
        except urllib.error.HTTPError as error:
            raise self._refused(error) from error
        except OSError as error:  # o URLError e a recusa da ligação são OSError
            raise _not_running(self.url) from error
        with response:
            try:
                for line in response:
                    if line.strip():
                        yield parse_chunk(_json_line(line))
            except TimeoutError as error:
                raise LanguageModelError(
                    f"The model {self._name} said nothing for more than {self._timeout:.0f} s. "
                    "On the first conversation Ollama is still loading it: try again."
                ) from error

    def _refused(self, error: urllib.error.HTTPError) -> LanguageModelError:
        """O erro de um pedido que o Ollama recusou, dito a quem usa o laboratório."""
        try:
            detail = str(json.loads(error.read() or b"{}").get("error", ""))
        except (ValueError, AttributeError):
            detail = ""
        finally:
            error.close()
        if error.code == 404:
            return LanguageModelError(
                f"Ollama doesn't have the model {self._name}: download it with "
                f"'ollama pull {self._name}' (only the first time)."
            )
        return LanguageModelError(f"Ollama refused the request: {detail or error.reason}.")


def chat_body(
    model: str,
    messages: Sequence[Message],
    *,
    think: bool,
    schema: Mapping[str, Any] | None,
    tools: Sequence[Tool],
    seed: int | None,
) -> dict[str, Any]:
    """O corpo do ``POST /api/chat``: a conversa, as opções, o formato e as ferramentas."""
    options: dict[str, Any] = {"temperature": TEMPERATURE}
    if seed is not None:
        options["seed"] = seed
    body: dict[str, Any] = {
        "model": model,
        "messages": [_message(message) for message in messages],
        "stream": True,
        "think": think,
        "options": options,
    }
    if schema is not None:
        body["format"] = dict(schema)
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for tool in tools
        ]
    return body


def parse_chunk(data: Mapping[str, Any]) -> Chunk:
    """Um pedaço da resposta, a partir de uma linha do NDJSON do Ollama.

    Raises:
        LanguageModelError: Se a linha trouxer um erro (o Ollama avisa a meio, assim).
    """
    if "error" in data:
        raise LanguageModelError(f"Ollama stopped the answer halfway: {data['error']}.")
    message = data.get("message") or {}
    calls = tuple(_tool_call(call) for call in message.get("tool_calls") or ())
    return Chunk(
        thinking=str(message.get("thinking") or ""),
        content=str(message.get("content") or ""),
        tool_calls=calls,
        done=bool(data.get("done", False)),
    )


def _message(message: Message) -> dict[str, Any]:
    written: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        written["tool_calls"] = [
            {"function": {"name": call.name, "arguments": dict(call.arguments)}}
            for call in message.tool_calls
        ]
    if message.tool_name is not None:
        written["tool_name"] = message.tool_name
    return written


def _tool_call(call: Mapping[str, Any]) -> ToolCall:
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):  # alguns modelos mandam os argumentos em texto
        arguments = json.loads(arguments or "{}")
    return ToolCall(str(function.get("name", "")), dict(arguments))


def _json_line(line: bytes) -> dict[str, Any]:
    try:
        data = json.loads(line)
    except ValueError as error:
        raise LanguageModelError(
            "Ollama answered a line that isn't JSON: is it really Ollama at that address?"
        ) from error
    if not isinstance(data, dict):
        raise LanguageModelError("Ollama answered unexpected JSON (it isn't an object).")
    return data


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    return _read_json(urllib.request.Request(url), timeout)


def _post_json(url: str, body: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _read_json(request, timeout)


def _read_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    url = request.full_url
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        error.close()
        raise LanguageModelError(f"Ollama answered {url} with error {error.code}.") from error
    except OSError as error:
        raise _not_running(url.rsplit("/api/", 1)[0]) from error
    except ValueError as error:
        raise LanguageModelError(
            f"The answer from {url} isn't JSON: is it really Ollama?"
        ) from error
    return data if isinstance(data, dict) else {}


def _not_running(url: str) -> LanguageModelError:
    return LanguageModelError(
        f"Ollama isn't answering at {url}: open the Ollama app, or run 'ollama serve' in a "
        "terminal."
    )
