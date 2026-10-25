"""Abre a API: ``uv run python -m ml_lab.api`` (a documentação fica em ``/docs``).

O endereço e a porta vêm das definições (``ML_LAB_API_HOST`` e ``ML_LAB_API_PORT``; por
omissão, http://127.0.0.1:8000, só neste computador).
"""

import uvicorn

from ml_lab.config import get_settings


def main() -> None:
    """Corre a API com o Uvicorn até se carregar em Ctrl+C."""
    settings = get_settings()
    uvicorn.run(
        "ml_lab.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
