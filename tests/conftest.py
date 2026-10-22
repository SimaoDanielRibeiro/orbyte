"""Configuração partilhada dos testes."""

import pytest

# Os testes destas pastas são mais lentos e correm antes de cada push. O marcador
# deixa excluí-los no ciclo rápido: uv run pytest -m "not integration and not api"
_FOLDER_MARKERS = {
    "integration": pytest.mark.integration,
    "api": pytest.mark.api,
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Marca cada teste de tests/integration e tests/api com o nome da pasta."""
    tests_dir = config.rootpath / "tests"
    for item in items:
        if not item.path.is_relative_to(tests_dir):
            continue
        marker = _FOLDER_MARKERS.get(item.path.relative_to(tests_dir).parts[0])
        if marker is not None:
            item.add_marker(marker)
