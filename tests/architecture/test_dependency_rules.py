"""Regras de dependência entre camadas (ver "Regra de dependências" em docs/architecture.md).

Cada ficheiro Python é lido com o módulo ``ast``, sem ser executado, e os seus imports
são comparados com as regras. Apanha imports no topo do ficheiro, dentro de funções e
dentro de ``if TYPE_CHECKING:``. Não apanha imports dinâmicos (``importlib``).
"""

import ast
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "ml_lab"
CORE = PACKAGE / "core"
FOUNDATION = CORE / "foundation"
DATA = CORE / "data"
SERVICES = CORE / "services"
TRACKING = CORE / "tracking"
API = PACKAGE / "api"

# Bibliotecas de interface, web e gráficos: o núcleo não depende de nenhuma delas.
INTERFACE_LIBRARIES = (
    "streamlit",
    "plotly",
    "matplotlib",
    "seaborn",
    "altair",
    "bokeh",
    "fastapi",
    "starlette",
    "uvicorn",
)


@dataclass(frozen=True)
class Import:
    """Um nome importado por um ficheiro, já com o nome completo (sem imports relativos)."""

    path: Path
    line: int
    name: str

    def __str__(self) -> str:
        return f"{self.path.relative_to(ROOT).as_posix()}:{self.line} importa {self.name}"


def is_within(name: str, prefix: str) -> bool:
    """Se ``name`` é ``prefix`` ou está dentro dele ("pandas.api" sim, "pandas_ta" não)."""
    return name == prefix or name.startswith(f"{prefix}.")


def python_files(*folders: Path, exclude: Iterable[Path] = ()) -> list[Path]:
    excluded = tuple(exclude)
    return sorted(
        path
        for folder in folders
        for path in folder.rglob("*.py")
        if not any(path.is_relative_to(other) for other in excluded)
    )


def module_name(path: Path) -> str:
    """Nome com que o ficheiro é importado (``ml_lab.core.x``)."""
    base = SRC if path.is_relative_to(SRC) else ROOT
    parts = path.relative_to(base).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def parse_imports(
    source: str | bytes, module: str, *, is_package: bool = False
) -> list[tuple[int, str]]:
    """Devolve ``(linha, nome completo)`` de cada nome importado em ``source``.

    ``from a.b import c`` conta como ``a.b.c``: ``c`` pode ser um submódulo. Com
    ``bytes``, a codificação é detetada como o Python faz (BOM incluído).
    """
    package = module if is_package else module.rpartition(".")[0]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # import relativo: sobe (level - 1) pacotes a partir do atual
                anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            found.extend(
                (node.lineno, base if alias.name == "*" else f"{base}.{alias.name}")
                for alias in node.names
            )
    return found


def find_violations(files: Iterable[Path], is_forbidden: Callable[[str], bool]) -> list[Import]:
    violations: list[Import] = []
    for path in files:
        # Lido em bytes: ficheiros gravados com BOM (comum no Windows) também são analisados.
        source = path.read_bytes()
        is_package = path.name == "__init__.py"
        for line, name in parse_imports(source, module_name(path), is_package=is_package):
            if is_forbidden(name):
                violations.append(Import(path, line, name))
    return violations


def report(rule: str, violations: list[Import]) -> str:
    return "\n".join([rule, *(f"  - {violation}" for violation in violations)])


# --- o próprio verificador ------------------------------------------------------------


def test_the_scanner_finds_the_project_files() -> None:
    # Sem ficheiros, todas as regras passariam sem verificar nada.
    assert python_files(FOUNDATION)
    assert python_files(DATA)
    assert python_files(API)


def test_the_scanner_sees_every_kind_of_import() -> None:
    source = (
        "import pandas as pd\n"
        "from typing import TYPE_CHECKING\n"
        "from ml_lab.core import services\n"
        "from . import sibling\n"
        "from ..data import boundary\n"
        "if TYPE_CHECKING:\n"
        "    import numpy\n"
        "def lazy() -> None:\n"
        "    import matplotlib\n"
    )

    names = {name for _, name in parse_imports(source, "ml_lab.core.models.base")}

    assert names == {
        "pandas",
        "typing.TYPE_CHECKING",
        "ml_lab.core.services",
        "ml_lab.core.models.sibling",
        "ml_lab.core.data.boundary",
        "numpy",
        "matplotlib",
    }


def test_the_scanner_reads_files_saved_with_a_bom() -> None:
    source = "﻿import pandas\n".encode()

    assert parse_imports(source, "ml_lab.core.models.base") == [(1, "pandas")]


def test_relative_imports_in_a_package_init_start_at_the_package() -> None:
    names = {
        name
        for _, name in parse_imports("from . import registry\n", "ml_lab.core", is_package=True)
    }

    assert names == {"ml_lab.core.registry"}


# --- regras ---------------------------------------------------------------------------


def test_foundation_only_depends_on_the_standard_library() -> None:
    def is_forbidden(name: str) -> bool:
        top_level = name.partition(".")[0]
        return not (
            top_level in sys.stdlib_module_names
            or top_level == "pydantic"
            or is_within(name, "ml_lab.core.foundation")
        )

    violations = find_violations(python_files(FOUNDATION), is_forbidden)

    assert not violations, report(
        "A fundação só importa a biblioteca padrão, o Pydantic e a própria fundação:", violations
    )


def test_core_does_not_import_the_api_or_interface_libraries() -> None:
    forbidden = ("ml_lab.api", *INTERFACE_LIBRARIES)

    violations = find_violations(
        python_files(CORE), lambda name: any(is_within(name, prefix) for prefix in forbidden)
    )

    assert not violations, report(
        "O núcleo não depende da API nem de bibliotecas de interface ou de gráficos:", violations
    )


def test_pandas_is_only_used_in_the_data_layer() -> None:
    violations = find_violations(
        python_files(PACKAGE, exclude=[DATA]), lambda name: is_within(name, "pandas")
    )

    assert not violations, report(
        "O pandas só pode ser importado em core/data/; o resto do sistema usa NumPy:", violations
    )


def test_the_database_and_the_saved_files_belong_to_the_tracking_layer() -> None:
    # O registo de experiências é trocável (regra 6 dos contratos): quem o usa só conhece o
    # ExperimentStore, nunca o SQLite nem o formato dos ficheiros.
    violations = find_violations(
        python_files(PACKAGE, exclude=[TRACKING]),
        lambda name: is_within(name, "sqlite3") or is_within(name, "joblib"),
    )

    assert not violations, report(
        "O SQLite e o joblib só se usam em core/tracking/, atrás do ExperimentStore:", violations
    )


def test_pytorch_is_only_used_by_the_neural_networks() -> None:
    # Tal como o pandas fica na camada de dados, o PyTorch fica nas redes: o resto do sistema
    # só conhece os contratos LabModel e Agent e recebe NumPy.
    neural = [
        CORE / "models" / "neural",
        CORE / "models" / "adapters" / "torch_adapter.py",
        CORE / "rl" / "agents" / "dqn.py",
    ]
    violations = find_violations(
        python_files(PACKAGE, exclude=neural), lambda name: is_within(name, "torch")
    )

    assert not violations, report(
        "O PyTorch só se usa nas redes: core/models/neural/, o adaptador torch_adapter.py e o "
        "DQN (core/rl/agents/dqn.py):",
        violations,
    )


def test_onnx_is_only_used_by_the_export() -> None:
    # O ONNX é uma saída para outras apps: só a exportação o conhece, e quem a usa recebe bytes.
    violations = find_violations(
        python_files(PACKAGE, exclude=[CORE / "export"]),
        lambda name: any(
            is_within(name, library) for library in ("onnx", "onnxruntime", "skl2onnx")
        ),
    )

    assert not violations, report(
        "O ONNX, o ONNX Runtime e o skl2onnx só se usam em core/export/:", violations
    )


def test_only_the_language_models_talk_to_the_network() -> None:
    # O laboratório corre todo no computador: o único pedido HTTP que o núcleo faz é ao
    # Ollama, e só o core/llm o faz (ADR 0042).
    violations = find_violations(
        python_files(PACKAGE, exclude=[CORE / "llm"]),
        lambda name: any(is_within(name, library) for library in ("urllib.request", "http.client")),
    )

    assert not violations, report(
        "Os pedidos HTTP (urllib.request, http.client) só se fazem em core/llm/:", violations
    )


def test_the_ollama_is_only_known_by_the_language_models_and_the_services() -> None:
    # O resto do laboratório conversa com a interface LanguageModel: trocar o Ollama por outro
    # modelo (ou pelo ScriptedModel, nos testes) não mexe em mais nada.
    violations = find_violations(
        python_files(PACKAGE, exclude=[CORE / "llm", SERVICES]),
        lambda name: is_within(name, "ml_lab.core.llm.ollama"),
    )

    assert not violations, report(
        "O cliente do Ollama só se importa em core/llm/ e nos serviços:", violations
    )


def test_the_core_does_not_import_the_services() -> None:
    violations = find_violations(
        python_files(CORE, exclude=[SERVICES]),
        lambda name: is_within(name, "ml_lab.core.services"),
    )

    assert not violations, report(
        "Os serviços são a porta de entrada do núcleo; o resto do núcleo não os importa:",
        violations,
    )


def test_the_api_only_reaches_the_core_through_services_and_schemas() -> None:
    # A interface web fala com o backend só pela API (em HTTP e WebSocket): a API é a única
    # porta para o núcleo, e só entra pelos serviços e pelos esquemas.
    allowed = ("ml_lab.core.services", "ml_lab.core.schemas")

    def is_forbidden(name: str) -> bool:
        return is_within(name, "ml_lab.core") and not any(is_within(name, a) for a in allowed)

    violations = find_violations(python_files(API), is_forbidden)

    assert not violations, report(
        "A API só chega ao núcleo através de core/services e core/schemas:", violations
    )
