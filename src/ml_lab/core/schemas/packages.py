"""O pedido de um pacote: o nome e as partes opcionais (ecrã Guardar, ADR 0039)."""

from pydantic import BaseModel, ConfigDict, Field

from ml_lab.core.data.serving import MAX_EXAMPLES


class PackageRequest(BaseModel):
    """O que vai num pacote, além das partes que vão sempre.

    Attributes:
        name: O nome do pacote (dá o nome do ficheiro); ``None`` usa o dataset e o modelo.
        include_model: Se leva o modelo treinado (sem ele, é "só a ficha": lê-se, mas não
            prevê, e não tem pickle nenhum).
        include_examples: Se leva linhas de exemplo; ``None`` aceita a sugestão do serviço
            (sim nos datasets de exemplo e nos sintéticos, não num ficheiro carregado).
        examples: Quantas linhas de exemplo.
        include_training: Se leva o treino passo a passo (para o rever depois de carregar).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = Field(default=None, max_length=120)
    include_model: bool = True
    include_examples: bool | None = None
    examples: int = Field(default=5, ge=1, le=MAX_EXAMPLES)
    include_training: bool = False
