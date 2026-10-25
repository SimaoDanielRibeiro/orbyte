"""O pedido de uma demonstração de um modelo (a página do modelo, no ecrã Modelos)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DemoRequest(BaseModel):
    """Que hiperparâmetros usar, e em que conjunto de exemplo.

    Attributes:
        hyperparameters: Os valores escolhidos; os que faltam ficam com o valor por omissão.
        example: O conjunto de exemplo (um dos ``examples`` do cartão do modelo); ``None``
            escolhe o primeiro da tarefa principal do modelo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    example: str | None = None
