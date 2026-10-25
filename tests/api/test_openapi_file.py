"""O contrato guardado para o frontend (``ui/web/openapi.json``) é o que a API serve.

O frontend gera os tipos TypeScript a partir do ficheiro, sem a API a correr (ADR 0026). Se
um serviço, uma rota ou um tipo mudar e o ficheiro não, os tipos do frontend mentem: este
teste falha e diz como o atualizar.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

OPENAPI_FILE = Path(__file__).resolve().parents[2] / "ui" / "web" / "openapi.json"


def test_the_saved_contract_is_the_one_the_api_serves(client: TestClient) -> None:
    saved = json.loads(OPENAPI_FILE.read_text(encoding="utf-8"))

    served = client.get("/openapi.json").json()

    assert saved == served, (
        "ui/web/openapi.json está desatualizado: corre "
        "`uv run python scripts/export_openapi.py` e revê a diferença no Git."
    )
