"""Serviço do catálogo de modelos: o que o ecrã Modelo mostra.

Cada cartão vem da ficha técnica do modelo (``ModelInfo``): nome, família,
implementação, capacidades, o JSON Schema dos hiperparâmetros (de onde as interfaces
geram o formulário) e o texto da ficha didática, inteiro e dividido pelas suas secções.

A página de cada modelo também o mostra a trabalhar: ``demo`` treina-o num conjunto de
exemplo pequeno, fora do histórico (``services/demos.py``, ADR 0028).
"""

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ml_lab.config import Settings, get_settings
from ml_lab.core.evaluation.explanations import METRIC_EXPLANATIONS, FormulaSymbol
from ml_lab.core.evaluation.metrics.base import METRICS, Metric
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, Implementation, LabModel, ModelFamily
from ml_lab.core.services.demos import DemoExample, ModelDemo, ModelDemos, examples_for
from ml_lab.core.services.laboratory import LaboratoryDemo, demos_for

DOCS_FOLDER: Final = Path("docs") / "algorithms"

INTRO: Final = "intro"
"""A chave do texto que vem antes da primeira secção da ficha, se houver."""

SUGGESTED_MODELS: Final[Mapping[TaskType, str]] = MappingProxyType(
    {
        TaskType.BINARY_CLASSIFICATION: "logistic_regression",
        TaskType.MULTICLASS_CLASSIFICATION: "logistic_regression",
        TaskType.REGRESSION: "linear_regression",
        TaskType.CLUSTERING: "kmeans",
        TaskType.DIMENSIONALITY_REDUCTION: "pca",
        TaskType.ANOMALY_DETECTION: "isolation_forest",
    }
)
"""O primeiro modelo a experimentar em cada tarefa: a referência que os outros têm de bater."""


@dataclass(frozen=True, slots=True)
class DocSection:
    """Uma secção da ficha didática.

    Attributes:
        key: Um nome estável, sem acentos (``intuicao``, ``matematica-essencial``): é com
            ele que a interface põe cada secção no seu lugar da página.
        title: O título, como está na ficha.
        text: O texto, em Markdown (com as fórmulas em LaTeX).
    """

    key: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Um modelo do catálogo, pronto a mostrar.

    Attributes:
        name: Identificador no registo.
        display_name: Nome a mostrar.
        summary: O que faz, numa frase.
        task_types: Tipos de tarefa que resolve.
        family: Família (para agrupar os cartões).
        implementation: Didática ou de uma biblioteca.
        iterative: Se treina por passos.
        supports_proba: Se dá probabilidades.
        hyperparameters_schema: JSON Schema dos hiperparâmetros.
        defaults: Os valores por omissão dos hiperparâmetros.
        doc: O texto da ficha didática, em Markdown (vazio se o ficheiro faltar).
        sections: A ficha dividida pelas suas secções (Intuição, Matemática essencial…).
        laboratory: As demonstrações do Laboratório que mostram como o modelo funciona.
        examples: Os conjuntos de exemplo da demonstração (``demo``), o de omissão primeiro.
        topics: Os temas do painel "Porquê?" que explicam o que a página do modelo mostra.
    """

    name: str
    display_name: str
    summary: str
    task_types: tuple[TaskType, ...]
    family: ModelFamily
    implementation: Implementation
    iterative: bool
    supports_proba: bool
    hyperparameters_schema: dict[str, Any]
    defaults: dict[str, Any]
    doc: str
    sections: tuple[DocSection, ...]
    laboratory: tuple[LaboratoryDemo, ...]
    examples: tuple[DemoExample, ...]
    topics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetricCard:
    """Uma métrica, pronta a explicar (o glossário e a explicação de cada métrica).

    Os valores de uma métrica num run vêm nos relatórios (``MetricResult``, com os termos
    da fórmula); o que não muda de run para run está aqui, e pede-se uma vez.

    Attributes:
        name: O identificador (o mesmo dos relatórios).
        display_name: O nome a mostrar.
        task_types: As tarefas em que se mede.
        higher_is_better: Se maior é melhor; ``None`` nas que só descrevem.
        as_percent: Se se mostra como percentagem.
        needs_proba: Se precisa das probabilidades (fica de fora nos modelos que não as dão).
        what: O que mede.
        how_to_read: Como se lê.
        formula: A definição, em LaTeX.
        symbols: O que quer dizer cada símbolo da fórmula.
    """

    name: str
    display_name: str
    task_types: tuple[TaskType, ...]
    higher_is_better: bool | None
    as_percent: bool
    needs_proba: bool
    what: str
    how_to_read: str
    formula: str
    symbols: tuple[FormulaSymbol, ...]


class CatalogService:
    """O catálogo de modelos.

    Args:
        settings: As definições (a raiz do projeto, onde estão as fichas didáticas).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._docs = (settings or get_settings()).home / DOCS_FOLDER
        self._demos = ModelDemos(settings)

    def models(self, task_type: TaskType | None = None) -> list[ModelCard]:
        """Os modelos (só os que resolvem ``task_type``, se for dado), por família e nome."""
        found = MODELS.for_task(task_type) if task_type is not None else dict(MODELS.items())
        cards = [self._card(model) for model in found.values()]
        return sorted(cards, key=lambda card: (card.family.order, card.display_name))

    def suggested_model(self, task_type: TaskType) -> str | None:
        """O primeiro modelo a experimentar para ``task_type`` (o mais simples que funciona bem)."""
        name = SUGGESTED_MODELS.get(task_type)
        return name if name in MODELS.for_task(task_type) else None

    def model(self, name: str) -> ModelCard:
        """O cartão de um modelo.

        Raises:
            ComponentNotFoundError: Se não existir (com uma sugestão do nome mais parecido).
        """
        return self._card(MODELS.get(name))

    def validate_hyperparameters(self, name: str, values: dict[str, Any]) -> dict[str, Any]:
        """Valida os hiperparâmetros e devolve-os completos (com os de omissão), em JSON.

        Raises:
            HyperparameterError: Com uma frase em português por cada valor recusado.
        """
        info = MODELS.get(name).info
        return info.hyperparameters.parse(values, model=name).model_dump(mode="json")

    def demo(
        self,
        name: str,
        hyperparameters: dict[str, Any] | None = None,
        example: str | None = None,
    ) -> ModelDemo:
        """O modelo a trabalhar num conjunto de exemplo pequeno, com estes hiperparâmetros.

        Não é um run: não vai para o histórico. A mesma pergunta duas vezes não volta a
        treinar (as últimas demonstrações ficam guardadas).

        Args:
            name: O modelo.
            hyperparameters: Os que mudam; os outros ficam com o valor por omissão.
            example: Um dos ``examples`` do cartão; ``None`` escolhe o primeiro.

        Raises:
            ComponentNotFoundError: Se o modelo não existir.
            HyperparameterError: Se os hiperparâmetros não forem válidos.
            InvalidParameterError: Se o conjunto de exemplo não servir para o modelo.
        """
        return self._demos.run(MODELS.get(name), hyperparameters or {}, example)

    def metrics(self, task_type: TaskType | None = None) -> list[MetricCard]:
        """As métricas (só as de ``task_type``, se for dada), com a fórmula e a explicação."""
        found = METRICS.for_task(task_type) if task_type is not None else dict(METRICS.items())
        return [_metric_card(metric) for metric in found.values()]

    def _card(self, model: type[LabModel[Any]]) -> ModelCard:
        info = model.info
        doc_path = self._docs / info.doc
        doc = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else ""
        laboratory = demos_for(info.name)
        return ModelCard(
            name=info.name,
            display_name=info.display_name,
            summary=info.summary,
            task_types=tuple(sorted(info.task_types)),
            family=info.family,
            implementation=info.implementation,
            iterative=info.iterative,
            supports_proba=info.supports_proba,
            hyperparameters_schema=info.hyperparameters.form_schema(),
            defaults=info.hyperparameters().model_dump(mode="json"),
            doc=doc,
            sections=doc_sections(doc),
            laboratory=laboratory,
            examples=examples_for(model),
            topics=_model_topics(info.task_types, info.family, info.iterative, laboratory),
        )


_CLASSIFICATION: Final = frozenset(
    {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
)
_UNSUPERVISED: Final = frozenset(
    {TaskType.CLUSTERING, TaskType.DIMENSIONALITY_REDUCTION, TaskType.ANOMALY_DETECTION}
)


def _model_topics(
    task_types: frozenset[TaskType],
    family: ModelFamily,
    iterative: bool,
    laboratory: tuple[LaboratoryDemo, ...],
) -> tuple[str, ...]:
    """Os temas do painel "Porquê?" que explicam a página de um modelo.

    São os do desenho do treino (a fronteira, os snapshots, a loss), os do que o modelo
    aprendeu (a rede, a árvore, a importância das features) e os das demonstrações do
    Laboratório que lhe dizem respeito.
    """
    topics: list[str] = []
    if task_types & _CLASSIFICATION:
        topics.append("decision_boundary")
    if task_types & _UNSUPERVISED:
        # A demonstração pinta os pontos com a coluna de referência, que não entra no treino.
        topics.append("reference")
    if TaskType.CLUSTERING in task_types:
        topics.append("silhouette")
    if TaskType.DIMENSIONALITY_REDUCTION in task_types:
        topics.append("projection")
    if iterative:
        topics += ["snapshots", "loss"]
    elif TaskType.REGRESSION in task_types:
        topics.append("loss")
    if family is ModelFamily.NEURAL:
        topics += ["gradient_descent", "neural_network", "backpropagation", "network_diagram"]
    elif iterative and family is ModelFamily.LINEAR:
        topics.append("gradient_descent")
    if family is ModelFamily.TREE:
        topics.append("tree_diagram")
    if family in (ModelFamily.TREE, ModelFamily.ENSEMBLE) and not (
        task_types & {TaskType.ANOMALY_DETECTION}
    ):
        topics.append("impurity_importance")
    topics += [demo.topic for demo in laboratory if demo.topic not in topics]
    return tuple(topics)


def _metric_card(metric: Metric) -> MetricCard:
    explanation = METRIC_EXPLANATIONS[metric.name]
    return MetricCard(
        name=metric.name,
        display_name=metric.display_name,
        task_types=tuple(sorted(metric.task_types)),
        higher_is_better=metric.higher_is_better,
        as_percent=metric.as_percent,
        needs_proba=metric.needs_proba,
        what=explanation.what,
        how_to_read=explanation.how_to_read,
        formula=explanation.formula,
        symbols=explanation.symbols,
    )


def doc_sections(doc: str) -> tuple[DocSection, ...]:
    """A ficha dividida pelos títulos de nível 2 (``## ``).

    O título da ficha (``# ``) é o nome do modelo, que o cartão já tem: fica de fora. O texto
    entre ele e a primeira secção, se houver, é a secção ``intro``.
    """
    sections: list[DocSection] = []
    title = ""
    lines: list[str] = []

    def close() -> None:
        text = "\n".join(lines).strip()
        if title or text:
            sections.append(DocSection(_slug(title) if title else INTRO, title, text))

    for line in doc.splitlines():
        if line.startswith("## "):
            close()
            title, lines = line[3:].strip(), []
        elif line.startswith("# ") and not title and not lines:
            continue  # o título da ficha
        else:
            lines.append(line)
    close()
    return tuple(sections)


def _slug(title: str) -> str:
    """O título sem acentos, em minúsculas e com hífenes.

    Por exemplo, ``Matemática essencial`` dá ``matematica-essencial``.
    """
    plain = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
