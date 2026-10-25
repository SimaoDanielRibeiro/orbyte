"""Guarda o OpenAPI da API em ``ui/web/openapi.json``: o contrato de onde o frontend gera os tipos.

O frontend (fase 10) não precisa da API a correr para conhecer o contrato: gera os tipos
TypeScript a partir deste ficheiro (por exemplo, com o ``openapi-typescript``). Como o
ficheiro está no Git, uma mudança no backend aparece como uma diferença no contrato, fácil
de rever. Um teste (``tests/api/test_openapi_file.py``) falha quando o ficheiro fica
desatualizado: corre este script depois de mudar um serviço, uma rota ou um tipo que a API
devolve (ADR 0026).

Uso, a partir da raiz do projeto::

    uv run python scripts/export_openapi.py
"""

import json
import sys
from pathlib import Path

from ml_lab.api.app import create_app
from ml_lab.config import get_settings
from ml_lab.core.services import Lab
from ml_lab.core.tracking.store import InMemoryStore
from ml_lab.logging import configure_logging, get_logger

logger = get_logger("scripts.export_openapi")

OPENAPI_FILE = Path("ui") / "web" / "openapi.json"
"""O ficheiro, relativo à raiz do projeto."""


def openapi_text() -> str:
    """O OpenAPI, como a API o serve em ``/openapi.json``, em JSON legível (e estável)."""
    # O registo em memória: gerar o esquema não deve criar nem tocar na base de dados.
    app = create_app(Lab.create(store=InMemoryStore()))
    return json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    """Escreve o ficheiro e diz onde ficou."""
    configure_logging()
    path = get_settings().home / OPENAPI_FILE
    path.write_text(openapi_text(), encoding="utf-8")
    logger.info("OpenAPI guardado", extra={"path": str(path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
