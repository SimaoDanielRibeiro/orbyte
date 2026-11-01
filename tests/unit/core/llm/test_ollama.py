"""O cliente do Ollama e o estado dos agentes de IA, contra um Ollama de mentira.

O Ollama de mentira é um servidor HTTP local que responde o que cada teste lhe disser: assim
testa-se o cliente a sério (o pedido, o NDJSON aos bocados, os erros) sem o Ollama instalado
nem um modelo descarregado.
"""

import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import LanguageModelError
from ml_lab.core.llm import Message, Tool, ToolCall, collect
from ml_lab.core.llm.ollama import OllamaModel, OllamaServer
from ml_lab.core.services.llm import LanguageModelService, has_model

SCHEMA = {"type": "object", "properties": {"max_depth": {"type": "integer"}}}


@dataclass
class FakeOllama:
    """O que o Ollama de mentira responde, e o que recebeu.

    Attributes:
        url: O endereço do servidor.
        version: A versão que diz ter.
        models: Os modelos que diz ter.
        chat_lines: As linhas do NDJSON da próxima conversa (cada uma, um objeto).
        chat_status: O código HTTP da conversa (404 para um modelo que não existe).
        chat_error: O erro da conversa, quando o código não é 200.
        requests: O corpo de cada ``POST /api/chat`` recebido.
    """

    url: str = ""
    version: str = "0.0-teste"
    models: list[str] = field(default_factory=list)
    chat_lines: list[dict[str, Any]] = field(default_factory=list)
    chat_status: int = 200
    chat_error: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)


def _handler(fake: FakeOllama) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            """Sem linhas no terminal a cada pedido."""

        def _json(self, status: int, data: dict[str, Any]) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/version":
                self._json(200, {"version": fake.version})
            elif self.path == "/api/tags":
                self._json(200, {"models": [{"name": name} for name in fake.models]})
            else:
                self._json(404, {"error": "não existe"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            fake.requests.append(json.loads(self.rfile.read(length)))
            if fake.chat_status != 200:
                self._json(fake.chat_status, {"error": fake.chat_error})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for line in fake.chat_lines:
                self.wfile.write(json.dumps(line).encode() + b"\n")
                self.wfile.flush()

    return Handler


@pytest.fixture
def fake_ollama() -> Iterator[FakeOllama]:
    """Um Ollama de mentira a correr, só durante o teste."""
    fake = FakeOllama()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def closed_url() -> str:
    """Um endereço onde nada escuta: a ligação é recusada, como sem o Ollama."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def line(*, thinking: str = "", content: str = "", done: bool = False) -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "thinking": thinking, "content": content},
        "done": done,
    }


# --- a conversa ----------------------------------------------------------------------


def test_the_thinking_arrives_apart_from_the_answer(fake_ollama: FakeOllama) -> None:
    fake_ollama.chat_lines = [
        line(thinking="A árvore "),
        line(thinking="decora o treino."),
        line(content='{"max_depth"'),
        line(content=": 4}"),
        line(done=True),
    ]
    model = OllamaModel(fake_ollama.url, "qwen3:8b")
    pieces: list[str] = []

    reply = collect(
        model.chat([Message("user", "Propõe.")], think=True, schema=SCHEMA, seed=7),
        on_chunk=lambda chunk: pieces.append(chunk.thinking),
    )

    assert reply.thinking == "A árvore decora o treino."
    assert json.loads(reply.content) == {"max_depth": 4}
    assert pieces[:2] == ["A árvore ", "decora o treino."]  # chega aos bocados, ao vivo


def test_the_request_asks_for_the_thinking_the_format_and_a_fixed_temperature(
    fake_ollama: FakeOllama,
) -> None:
    fake_ollama.chat_lines = [line(content="{}", done=True)]
    model = OllamaModel(fake_ollama.url, "qwen3:8b")

    collect(model.chat([Message("user", "Propõe.")], think=True, schema=SCHEMA, seed=7))

    body = fake_ollama.requests[0]
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is True
    assert body["think"] is True
    assert body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0.0, "seed": 7}
    assert body["messages"] == [{"role": "user", "content": "Propõe."}]


def test_the_tools_go_out_and_the_calls_come_back(fake_ollama: FakeOllama) -> None:
    call = {"function": {"name": "show_candidate", "arguments": {"number": 3}}}
    fake_ollama.chat_lines = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": True}
    ]
    tool = Tool(
        "show_candidate",
        "Mostra um candidato do quadro.",
        {"type": "object", "properties": {"number": {"type": "integer"}}},
    )
    history = [
        Message("user", "Porque ganhou o 3?"),
        Message("assistant", "", (ToolCall("list_candidates", {}),)),
        Message("tool", "[1, 2, 3]", tool_name="list_candidates"),
    ]

    reply = collect(OllamaModel(fake_ollama.url, "qwen3:8b").chat(history, tools=[tool]))

    assert reply.tool_calls == (ToolCall("show_candidate", {"number": 3}),)
    body = fake_ollama.requests[0]
    assert body["tools"][0]["function"]["name"] == "show_candidate"
    assert body["messages"][1]["tool_calls"] == [
        {"function": {"name": "list_candidates", "arguments": {}}}
    ]
    assert body["messages"][2]["tool_name"] == "list_candidates"


def test_a_missing_model_says_how_to_download_it(fake_ollama: FakeOllama) -> None:
    fake_ollama.chat_status = 404
    fake_ollama.chat_error = "model 'qwen3:8b' not found"

    with pytest.raises(LanguageModelError, match="ollama pull qwen3:8b"):
        collect(OllamaModel(fake_ollama.url, "qwen3:8b").chat([Message("user", "Olá")]))


def test_an_error_in_the_middle_of_the_answer_stops_it(fake_ollama: FakeOllama) -> None:
    fake_ollama.chat_lines = [line(content="Come"), {"error": "sem memória na gráfica"}]

    with pytest.raises(LanguageModelError, match="sem memória na gráfica"):
        collect(OllamaModel(fake_ollama.url, "qwen3:8b").chat([Message("user", "Olá")]))


def test_without_the_ollama_the_error_says_how_to_start_it(closed_url: str) -> None:
    model = OllamaModel(closed_url, "qwen3:8b", timeout=2)

    with pytest.raises(LanguageModelError, match="ollama serve"):
        collect(model.chat([Message("user", "Olá")]))


def test_the_server_says_its_version_and_its_models(fake_ollama: FakeOllama) -> None:
    fake_ollama.models = ["qwen3:8b", "llama3.1:8b"]

    info = OllamaServer(fake_ollama.url).info()

    assert info.version == "0.0-teste"
    assert info.models == ("llama3.1:8b", "qwen3:8b")


# --- o estado, no serviço --------------------------------------------------------------


def status_service(url: str, tmp_path: Path, model: str = "qwen3:8b") -> LanguageModelService:
    settings = Settings.from_env(
        {"ML_LAB_OLLAMA_URL": url, "ML_LAB_OLLAMA_MODEL": model}, home=tmp_path
    )
    return LanguageModelService(settings)


def test_without_the_ollama_the_status_says_how_to_start_it(
    closed_url: str, tmp_path: Path
) -> None:
    status = status_service(closed_url, tmp_path).status()

    assert not status.ready
    assert not status.running
    assert [notice.code for notice in status.notices] == ["ollama_not_running"]
    assert "ollama pull qwen3:8b" in status.notices[0].message


def test_without_the_model_the_status_says_how_to_download_it(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.models = ["llama3.1:8b"]

    status = status_service(fake_ollama.url, tmp_path).status()

    assert status.running
    assert not status.ready
    assert status.models == ("llama3.1:8b",)
    assert [notice.code for notice in status.notices] == ["model_not_installed"]
    assert "llama3.1:8b" in status.notices[0].message


def test_with_the_model_the_ai_agents_can_work(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.models = ["qwen3:8b"]

    status = status_service(fake_ollama.url, tmp_path).status()

    assert status.ready
    assert status.version == "0.0-teste"
    assert status.notices == ()


def test_a_model_without_a_tag_is_the_latest() -> None:
    assert has_model(("qwen3:latest",), "qwen3")
    assert has_model(("qwen3:8b",), "qwen3:8b")
    assert not has_model(("qwen3:4b",), "qwen3:8b")
    assert not has_model(("qwen3:8b",), "qwen3")
