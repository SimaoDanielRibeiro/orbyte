"""Configuração partilhada dos testes."""

from collections.abc import Iterator

import pytest

from ml_lab.config import get_settings
from ml_lab.process import run_at_full_speed

# Os testes destas pastas são mais lentos e correm antes de cada push. O marcador
# deixa excluí-los no ciclo rápido: uv run pytest -m "not integration and not api"
_FOLDER_MARKERS = {
    "integration": pytest.mark.integration,
    "api": pytest.mark.api,
}


def pytest_configure(config: pytest.Config) -> None:
    """Pede ao Windows que não trave o processo dos testes (ver ``ml_lab.process``)."""
    run_at_full_speed()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Marca cada teste de tests/integration e tests/api com o nome da pasta."""
    tests_dir = config.rootpath / "tests"
    for item in items:
        if not item.path.is_relative_to(tests_dir):
            continue
        marker = _FOLDER_MARKERS.get(item.path.relative_to(tests_dir).parts[0])
        if marker is not None:
            item.add_marker(marker)


@pytest.fixture(autouse=True, scope="session")
def _isolated_folders(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Os testes nunca escrevem na base de dados, nos artefactos nem nos uploads verdadeiros.

    Cada ``Lab.create()`` sem definições explícitas lê o ambiente; aqui, o ambiente aponta
    para uma pasta temporária, a mesma em toda a sessão de testes.
    """
    folder = tmp_path_factory.mktemp("ml_lab")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ML_LAB_UPLOADS_DIR", str(folder / "uploads"))
        patch.setenv("ML_LAB_ARTIFACTS_DIR", str(folder / "artifacts"))
        patch.setenv("ML_LAB_DATABASE_PATH", str(folder / "artifacts" / "ml_lab.db"))
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()
