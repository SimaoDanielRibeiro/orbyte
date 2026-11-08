"""Arranca a API e a interface web juntas: o comando para trabalhar na interface.

Uso, a partir da raiz do projeto::

    uv run python scripts/dev.py               # e abre o browser quando estiver pronto
    uv run python scripts/dev.py --no-browser  # sem abrir o browser

Corre a API (``python -m ml_lab.api``) e o servidor de desenvolvimento do Vite (``npm run dev``,
em ``ui/web``), cada um no seu processo, com a saída dos dois neste terminal. Quando os dois
respondem, diz "Pronto" e abre a interface (http://localhost:5173) no browser: abrir a página
antes de a API acabar de arrancar mostrava um erro. A interface recarrega sozinha ao gravar um
ficheiro. ``Ctrl+C`` pára os dois; se um deles parar sozinho, o outro pára também.

É o comando de quem acabou de descarregar o projeto, em qualquer computador (ADR 0054), por isso
verifica antes de arrancar o que costuma falhar, e diz o que fazer:

- o Node.js tem de existir e ser o 24 ou mais recente (o ``engines`` do ``package.json``);
- as portas da API e da interface têm de estar livres (uma API esquecida noutro terminal é o
  caso mais comum);
- as dependências da interface instalam-se com ``npm ci`` (as versões exatas do
  ``package-lock.json``) na primeira vez e sempre que o ``package-lock.json`` mudar (um
  ``git pull``). As do backend, o ``uv run`` já as acerta sozinho com o ``uv.lock``.
"""

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_lab.config import get_settings
from ml_lab.logging import configure_logging, get_logger

logger = get_logger("scripts.dev")

WEB = Path("ui") / "web"
"""A pasta da interface web, relativa à raiz do projeto."""

WEB_PORT = 5173
"""A porta do Vite (``server.port`` no ``ui/web/vite.config.ts``)."""

WEB_URL = f"http://localhost:{WEB_PORT}"
NODE_MAJOR = 24
"""A versão mínima do Node.js (o ``engines`` do ``ui/web/package.json``)."""

POLL_SECONDS = 0.5
STOP_SECONDS = 10
PORT_TIMEOUT = 0.3
"""Quanto se espera para saber se alguém já escuta numa porta, em segundos."""

READY_TIMEOUT = 1.0
"""Quanto se espera por cada resposta, ao perguntar se a API e a interface já estão prontas."""


def main(argv: Sequence[str] | None = None) -> int:
    """Corre a API e a interface até um deles parar (ou até ``Ctrl+C``)."""
    parser = argparse.ArgumentParser(description="Arranca a API e a interface do Orbyte.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="não abre a interface no browser quando estiver pronta",
    )
    options = parser.parse_args(argv)
    configure_logging()
    settings = get_settings()
    web = settings.home / WEB
    npm = shutil.which("npm")
    node = shutil.which("node")
    if npm is None or node is None:
        logger.error(
            "Falta o Node.js (que traz o npm): instala o %s ou mais recente, de "
            "https://nodejs.org, abre um terminal novo e corre outra vez.",
            NODE_MAJOR,
        )
        return 1
    major = node_major(node)
    if major is not None and major < NODE_MAJOR:
        logger.error(
            "O Node.js deste computador é o %s, e a interface precisa do %s ou mais recente: "
            "instala-o de https://nodejs.org e corre outra vez.",
            major,
            NODE_MAJOR,
        )
        return 1
    host = "127.0.0.1" if settings.api_host in ("0.0.0.0", "::") else settings.api_host
    for name, address, port, change in (
        ("API", host, settings.api_port, "muda a porta no .env (ML_LAB_API_PORT)"),
        ("interface", "localhost", WEB_PORT, "fecha o programa que a usa"),
    ):
        if port_in_use(address, port):
            logger.error(
                "A porta %s (da %s) já está ocupada: talvez o Orbyte já esteja a correr noutro "
                "terminal. Fecha-o, ou %s, e corre outra vez.",
                port,
                name,
                change,
            )
            return 1
    if needs_install(web):
        logger.info("A instalar as dependências da interface, com as versões do package-lock.json")
        if subprocess.run([npm, "ci"], cwd=web, check=False).returncode != 0:
            logger.error("O npm ci falhou: as mensagens acima dizem porquê.")
            return 1
    processes = {
        "API": _start([sys.executable, "-m", "ml_lab.api"], settings.home),
        "interface": _start([npm, "run", "dev"], web),
    }
    logger.info(
        "A API e a interface estão a arrancar (Ctrl+C pára as duas)",
        extra={"interface": WEB_URL, "api": f"http://{settings.api_host}:{settings.api_port}"},
    )
    health = f"http://{host}:{settings.api_port}/health"
    ready = False
    try:
        while True:
            for name, process in processes.items():
                code = process.poll()
                if code is not None:
                    logger.error(
                        "Um dos processos parou sozinho: paro o outro",
                        extra={"processo": name, "codigo": code},
                    )
                    return code or 1
            if not ready and responds(health) and responds(WEB_URL):
                ready = True
                logger.info("Pronto: o Orbyte está em %s (Ctrl+C pára tudo)", WEB_URL)
                if not options.no_browser and can_open_browser():
                    webbrowser.open(WEB_URL)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        logger.info("A parar a API e a interface")
        return 0
    finally:
        for process in processes.values():
            _stop(process)


def node_major(node: str) -> int | None:
    """A versão principal do Node.js instalado, ou ``None`` se não se souber."""
    try:
        found = subprocess.run(
            [node, "--version"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_node_major(found.stdout)


def parse_node_major(version: str) -> int | None:
    """A versão principal no que ``node --version`` escreve: ``24`` em ``v24.11.0``."""
    match = re.match(r"v?(\d+)\.", version.strip())
    return int(match.group(1)) if match else None


def responds(url: str) -> bool:
    """Se o endereço já responde (a API à pergunta ``/health``, o Vite com a página)."""
    try:
        with urllib.request.urlopen(url, timeout=READY_TIMEOUT) as response:
            return bool(200 <= response.status < 400)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def can_open_browser(platform: str = sys.platform, environ: Mapping[str, str] = os.environ) -> bool:
    """Se há um ecrã onde abrir o browser.

    No Windows e no macOS há sempre. No Linux, só com uma sessão gráfica (``DISPLAY`` ou
    ``WAYLAND_DISPLAY``): sem ela, o ``webbrowser`` abria um browser de texto (o ``lynx``, o
    ``w3m``) dentro deste terminal, por cima dos registos.
    """
    if platform.startswith("linux"):
        return bool(environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"))
    return True


def port_in_use(host: str, port: int) -> bool:
    """Se já há alguém a escutar em ``host:port`` (todas as moradas do nome, IPv4 e IPv6)."""
    try:
        with socket.create_connection((host, port), timeout=PORT_TIMEOUT):
            return True
    except OSError:
        return False


def needs_install(web: Path) -> bool:
    """Se faltam as dependências da interface, ou se o ``package-lock.json`` mudou depois delas.

    O npm deixa em ``node_modules/.package-lock.json`` o retrato do que instalou: se o
    ``package-lock.json`` do projeto é mais recente do que ele (um ``git pull`` trouxe
    dependências novas), as que estão instaladas já não são as certas.
    """
    installed = web / "node_modules" / ".package-lock.json"
    if not installed.is_file():
        return True
    return (web / "package-lock.json").stat().st_mtime > installed.stat().st_mtime


def _start(command: list[str], folder: Path) -> subprocess.Popen[bytes]:
    """Arranca um processo; fora do Windows, num grupo só seu, para o parar com os filhos."""
    if sys.platform == "win32":
        process = subprocess.Popen(command, cwd=folder)
    else:
        process = subprocess.Popen(command, cwd=folder, start_new_session=True)
    return process


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Pára um processo e os que ele lançou (o npm lança o Vite: sozinho, ficava órfão)."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=STOP_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()


if __name__ == "__main__":
    sys.exit(main())
