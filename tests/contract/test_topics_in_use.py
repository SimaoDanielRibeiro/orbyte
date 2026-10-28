"""Cada tema que a interface pede existe no backend.

Os textos didáticos escrevem-se uma só vez, no backend (regra 5 do ``CONTRIBUTING.md``), e a
interface pede-os pelo nome: o painel "Porquê?" e o "como ler" por baixo de cada gráfico.
Um nome que deixe de existir não dá erro na interface (o texto só não aparece), por isso
este teste procura os nomes no código dela.

Os temas pedem-se assim: ``<WhyButton topics={["split", …]} />``,
``<TopicText topic="split" />``, ``useTopic("split")`` ou ``topic: "split"`` (num objeto).
"""

import re
from pathlib import Path

from ml_lab.core.services import CatalogService, topics

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "ui" / "web" / "src"

_NAME = re.compile(r"[\"']([a-z0-9_]+)[\"']")
_WEB_SINGLE = re.compile(r"(?:topic=|useTopic\(|topic: )\"([a-z0-9_]+)\"")
_WEB_LIST = re.compile(r"topics=\{\[([^\]]*)\]\}")


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


def test_the_search_finds_the_topics_of_the_interface() -> None:
    # Se as expressões deixarem de encontrar os nomes, o teste de baixo passaria sem verificar
    # nada: estes são pedidos de certeza (o ecrã Dados, o Preparar e o Explicar).
    assert {"input", "validation", "lineage", "importance_bars"} <= _web_topics()


def test_every_topic_the_interface_asks_for_exists() -> None:
    missing = sorted(_web_topics() - topics().keys())

    assert not missing, f"A interface pede temas que não existem: {missing}"


def test_every_model_page_explains_itself_with_existing_topics() -> None:
    cards = CatalogService().models()

    missing = {card.name: set(card.topics) - topics().keys() for card in cards}

    assert all(card.topics for card in cards), "Cada página de modelo tem o seu «Porquê?»."
    assert not any(missing.values()), f"Temas que não existem: {missing}"
