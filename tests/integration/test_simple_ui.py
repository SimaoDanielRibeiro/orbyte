"""A interface simples abre, e cada ecrã do plano abre sem erros (critério da fase 0)."""

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI_DIR = Path(__file__).resolve().parents[2] / "ui" / "simple"
PAGES = sorted((UI_DIR / "pages").glob("*.py"))


def open_app() -> AppTest:
    return AppTest.from_file(UI_DIR / "app.py", default_timeout=30).run()


def test_app_opens_on_the_data_screen() -> None:
    app = open_app()

    assert not app.exception
    assert app.title[0].value == "Dados"


def test_sidebar_shows_the_current_experiment() -> None:
    app = open_app()

    assert app.sidebar.subheader[0].value == "Experiência atual"
    assert app.sidebar.button[0].disabled  # guardar experiências chega na fase 5


def test_there_is_one_page_per_screen_of_the_plan() -> None:
    assert len(PAGES) == 11


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.stem)
def test_every_screen_opens_without_errors(page: Path) -> None:
    app = open_app()

    app.switch_page(page.relative_to(UI_DIR).as_posix()).run()

    assert not app.exception
    assert len(app.title) == 1


@pytest.mark.parametrize("page", [UI_DIR / "app.py", *PAGES], ids=lambda page: page.stem)
def test_screens_have_no_loose_text(page: Path) -> None:
    # A "magia" do Streamlit desenha qualquer expressão solta de um ecrã, incluindo uma
    # docstring por baixo de uma constante. Só a docstring do módulo é ignorada.
    body = ast.parse(page.read_bytes()).body
    loose = [
        node.lineno
        for node in body[1:]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    ]

    assert not loose, f"{page.name}: texto solto nas linhas {loose} (usa um comentário)"
