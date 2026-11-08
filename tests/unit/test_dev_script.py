"""O arranque do dia a dia (``scripts/dev.py``): o que verifica antes de arrancar (ADR 0054).

É o primeiro comando de quem descarrega o projeto noutro computador, por isso diz o que fazer
quando falta alguma coisa: o Node.js antigo, uma porta ocupada, as dependências da interface por
instalar ou desatualizadas depois de um ``git pull``.
"""

import importlib.util
import os
import socket
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
    ("written", "major"),
    [("v24.11.0\n", 24), ("v18.20.4", 18), ("26.0.0", 26), ("", None), ("node", None)],
)
def test_the_node_version_is_read_as_node_writes_it(
    dev: ModuleType, written: str, major: int | None
) -> None:
    assert dev.parse_node_major(written) == major


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
