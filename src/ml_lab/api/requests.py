"""Os corpos dos pedidos que juntam vários valores (os outros vão no caminho ou na query).

Quando um método de um serviço recebe uma só configuração (``ExperimentConfig``,
``SourceSpec``), a rota recebe-a tal como está. Estes modelos existem só para os métodos que
recebem vários valores ao mesmo tempo.
"""

from pydantic import BaseModel, ConfigDict, Field

from ml_lab.core.services import PreparationConfig


class GenerateRequest(BaseModel):
    """Gerar um dataset sintético.

    Attributes:
        generator: O gerador (``moons``, ``blobs``...).
        parameters: Os parâmetros do gerador; os que faltam ficam com o valor por omissão.
        seed: A seed: a mesma seed dá os mesmos pontos.
    """

    model_config = ConfigDict(extra="forbid")

    generator: str = Field(min_length=1)
    parameters: dict[str, float] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0)


class PrepareRequest(BaseModel):
    """Preparar um dataset carregado.

    Attributes:
        config: O alvo, a tarefa, a limpeza, a divisão, as transformações e a projeção.
        seed: A seed da divisão (a mesma do treino, para os dados serem os mesmos).
    """

    model_config = ConfigDict(extra="forbid")

    config: PreparationConfig = Field(default_factory=PreparationConfig)
    seed: int = Field(default=42, ge=0)


class SaveRequest(BaseModel):
    """Guardar um run (com um nome novo, se se quiser)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None


class RenameRequest(BaseModel):
    """Mudar o nome de um run."""

    model_config = ConfigDict(extra="forbid")

    name: str


class CompareRequest(BaseModel):
    """Comparar runs lado a lado (de 2 a 5)."""

    model_config = ConfigDict(extra="forbid")

    run_ids: list[str]
