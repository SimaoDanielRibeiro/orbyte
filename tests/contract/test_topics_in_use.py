"""Cada tema que as interfaces pedem existe no backend.

Os textos didáticos escrevem-se uma só vez, no backend (regra 5 do ``CONTRIBUTING.md``), e as
interfaces pedem-nos pelo nome: o painel "Porquê?" e o "como ler" por baixo de cada gráfico.
Um nome que deixe de existir não dá erro na interface web (o texto só não aparece), por isso
este teste procura os nomes no código das duas interfaces.

Na interface web, os temas pedem-se assim: ``<WhyButton topics={["split", …]} />``,
``<TopicText topic="split" />``, ``useTopic("split")`` ou ``topic: "split"`` (num objeto).
Na simples, com ``render_why("split", …)`` ou ``explain("split")``.
"""

import re
from pathlib import Path

import pytest

from ml_lab.core.services import CatalogService, topics

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "ui" / "web" / "src"
SIMPLE = ROOT / "ui" / "simple"

_NAME = re.compile(r"[\"']([a-z0-9_]+)[\"']")
_WEB_SINGLE = re.compile(r"(?:topic=|useTopic\(|topic: )\"([a-z0-9_]+)\"")
_WEB_LIST = re.compile(r"topics=\{\[([^\]]*)\]\}")
_SIMPLE_CALL = re.compile(r"\b(?:render_why|explain)\(([^)]*)\)")


def _web_topics() -> set[str]:
    found: set[str] = set()
    for path in WEB.rglob("*.tsx"):
        if "generated" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        found.update(_WEB_SINGLE.findall(text))
        for names in _WEB_LIST.findall(text):
            found.update(_NAME.findall(names))
    return found


def _simple_topics() -> set[str]:
    found: set[str] = set()
    for path in SIMPLE.rglob("*.py"):
        for arguments in _SIMPLE_CALL.findall(path.read_text(encoding="utf-8")):
            found.update(_NAME.findall(arguments))
    return found


def test_the_search_finds_the_topics_of_both_interfaces() -> None:
    # Se as expressões deixarem de encontrar os nomes, o teste de baixo passaria sem verificar
    # nada: estes dois são pedidos de certeza (o ecrã Dados e o Preparar).
    assert {"input", "validation"} <= _web_topics()
    assert {"split", "lineage"} <= _simple_topics()


@pytest.mark.parametrize(
    ("interface", "names"),
    [("web", _web_topics()), ("simples", _simple_topics())],
)
def test_every_topic_an_interface_asks_for_exists(interface: str, names: set[str]) -> None:
    missing = sorted(names - topics().keys())

    assert not missing, f"A interface {interface} pede temas que não existem: {missing}"


def test_every_model_page_explains_itself_with_existing_topics() -> None:
    cards = CatalogService().models()

    missing = {card.name: set(card.topics) - topics().keys() for card in cards}

    assert all(card.topics for card in cards), "Cada página de modelo tem o seu «Porquê?»."
    assert not any(missing.values()), f"Temas que não existem: {missing}"
