"""O arranque do dia a dia (``scripts/dev.py``): o que verifica antes de arrancar (ADR 0054).

É o primeiro comando de quem descarrega o projeto noutro computador, por isso diz o que fazer
quando falta alguma coisa: o Node.js antigo, uma porta ocupada, as dependências da interface por
instalar ou desatualizadas depois de um ``git pull``.
"""

import importlib.util
import os
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev.py"


@pytest.fixture(scope="module")
def dev() -> ModuleType:
    """O script, carregado como um módulo (a pasta ``scripts`` não é um pacote)."""
    spec = importlib.util.spec_from_file_location("dev_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("written", "version"),
    [
        ("v24.11.0\n", (24, 11)),
        ("v18.20.4", (18, 20)),
        ("26.0.0", (26, 0)),
        ("v24", None),
        ("", None),
        ("node", None),
    ],
)
def test_the_node_version_is_read_as_node_writes_it(
    dev: ModuleType, written: str, version: tuple[int, int] | None
) -> None:
    assert dev.parse_node_version(written) == version


@pytest.mark.parametrize(
    ("written", "recent"),
    [
        ("v24.15.0", False),  # vai abaixo sem aviso no Windows, com muitas ligações (ADR 0056)
        ("v24.9.1", False),  # 9 vem antes de 16: compara-se como números, não como texto
        ("v22.20.0", False),
        ("v24.16.0", True),
        ("v24.22.2", True),
        ("v26.0.0", True),
    ],
)
def test_the_node_version_must_be_at_least_the_minimum(
    dev: ModuleType, written: str, recent: bool
) -> None:
    version = dev.parse_node_version(written)

    assert version is not None
    assert (version >= dev.NODE_MINIMUM) is recent


class _Health(BaseHTTPRequestHandler):
    """Responde como a API: 200 em ``/health`` e 404 no resto."""

    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Sem registos no terminal dos testes."""


@pytest.fixture
def server() -> Iterator[str]:
    """Um servidor HTTP local, numa porta livre; dá o endereço."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_it_is_ready_only_when_the_address_answers(dev: ModuleType, server: str) -> None:
    assert dev.responds(f"{server}/health") is True
    assert dev.responds(f"{server}/nada") is False  # responde, mas com um erro
    port = int(server.rsplit(":", 1)[1])
    with socket.socket() as probe:  # uma porta livre, onde ninguém escuta
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert free != port
    assert dev.responds(f"http://127.0.0.1:{free}/health") is False


@pytest.mark.parametrize(
    ("platform", "environ", "opens"),
    [
        ("win32", {}, True),
        ("darwin", {}, True),
        ("linux", {"DISPLAY": ":0"}, True),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, True),
        ("linux", {}, False),  # por SSH, sem ecrã: abria um browser de texto no terminal
    ],
)
def test_the_browser_only_opens_where_there_is_a_screen(
    dev: ModuleType, platform: str, environ: dict[str, str], opens: bool
) -> None:
    assert dev.can_open_browser(platform, environ) is opens


def test_a_port_with_someone_listening_is_in_use(dev: ModuleType) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]

        assert dev.port_in_use("127.0.0.1", port) is True
    assert dev.port_in_use("127.0.0.1", port) is False


def test_the_interface_installs_the_first_time_and_after_the_lockfile_changes(
    dev: ModuleType, tmp_path: Path
) -> None:
    lockfile = tmp_path / "package-lock.json"
    lockfile.write_text("{}", encoding="utf-8")
    assert dev.needs_install(tmp_path) is True  # nada instalado

    installed = tmp_path / "node_modules" / ".package-lock.json"
    installed.parent.mkdir()
    installed.write_text("{}", encoding="utf-8")
    os.utime(lockfile, (1_000, 1_000))
    os.utime(installed, (2_000, 2_000))
    assert dev.needs_install(tmp_path) is False  # instalado depois do lockfile

    os.utime(lockfile, (3_000, 3_000))  # um git pull trouxe um lockfile novo
    assert dev.needs_install(tmp_path) is True
