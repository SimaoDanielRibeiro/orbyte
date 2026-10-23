"""Serviço do catálogo de modelos: o que o ecrã Modelo mostra.

Cada cartão vem da ficha técnica do modelo (``ModelInfo``): nome, família,
implementação, capacidades, o JSON Schema dos hiperparâmetros (de onde as interfaces
geram o formulário) e o texto da ficha didática.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ml_lab.config import Settings, get_settings
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, Implementation, LabModel, ModelFamily

DOCS_FOLDER: Final = Path("docs") / "algorithms"


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


class CatalogService:
    """O catálogo de modelos.

    Args:
        settings: As definições (a raiz do projeto, onde estão as fichas didáticas).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._docs = (settings or get_settings()).home / DOCS_FOLDER

    def models(self, task_type: TaskType | None = None) -> list[ModelCard]:
        """Os modelos (só os que resolvem ``task_type``, se for dado), pelo nome a mostrar."""
        found = MODELS.for_task(task_type) if task_type is not None else dict(MODELS.items())
        cards = [self._card(model) for model in found.values()]
        return sorted(cards, key=lambda card: (card.family, card.display_name))

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

    def _card(self, model: type[LabModel[Any]]) -> ModelCard:
        info = model.info
        doc_path = self._docs / info.doc
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
            doc=doc_path.read_text(encoding="utf-8") if doc_path.is_file() else "",
        )
