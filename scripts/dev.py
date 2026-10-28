"""Arranca a API e a interface web juntas: o comando para trabalhar na interface.

Uso, a partir da raiz do projeto::

    uv run python scripts/dev.py

Corre a API (``python -m ml_lab.api``) e o servidor de desenvolvimento do Vite (``npm run dev``,
em ``ui/web``), cada um no seu processo, com a saída dos dois neste terminal. A interface abre
em http://localhost:5173 e recarrega sozinha ao gravar um ficheiro. ``Ctrl+C`` pára os dois; se
um deles parar sozinho (a porta já estava ocupada, por exemplo), o outro pára também.

Na primeira vez, se faltarem as dependências da interface (``ui/web/node_modules``), instala-as
com ``npm install``.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from ml_lab.config import get_settings
from ml_lab.logging import configure_logging, get_logger

logger = get_logger("scripts.dev")

WEB = Path("ui") / "web"
"""A pasta da interface web, relativa à raiz do projeto."""

WEB_URL = "http://localhost:5173"
POLL_SECONDS = 0.5
STOP_SECONDS = 10


def main() -> int:
    """Corre a API e a interface até um deles parar (ou até ``Ctrl+C``)."""
    configure_logging()
    settings = get_settings()
    web = settings.home / WEB
    npm = shutil.which("npm")
    if npm is None:
        logger.error("Falta o npm: instala o Node 24 (que o traz) e corre outra vez.")
        return 1
    if not (web / "node_modules").is_dir():
        logger.info("A instalar as dependências da interface (só da primeira vez)")
        if subprocess.run([npm, "install"], cwd=web, check=False).returncode != 0:
            logger.error("O npm install falhou: as mensagens acima dizem porquê.")
            return 1
    processes = {
        "API": _start([sys.executable, "-m", "ml_lab.api"], settings.home),
        "interface": _start([npm, "run", "dev"], web),
    }
    logger.info(
        "A API e a interface estão a arrancar (Ctrl+C pára as duas)",
        extra={"interface": WEB_URL, "api": f"http://{settings.api_host}:{settings.api_port}"},
    )
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
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        logger.info("A parar a API e a interface")
        return 0
    finally:
        for process in processes.values():
            _stop(process)


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
