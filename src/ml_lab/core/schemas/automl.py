"""O pedido aos agentes (fase 9): os dados, o alvo, a métrica e o tamanho da procura (Pydantic).

É o que entra no serviço dos agentes e o que fica no resultado, com a seed já resolvida: com
o mesmo pedido, a procura repete-se tal e qual. Vai e volta de JSON sem perder nada, por isso
serve também a API.

Exemplo:
    >>> request = AutoMLRequest.model_validate(
    ...     {
    ...         "dataset": {"kind": "sample", "params": {"name": "titanic"}},
    ...         "target": "Survived",
    ...         "settings": {"budget": 12, "folds": 3},
    ...     }
    ... )
    >>> request.settings.finalists
    3
    >>> AutoMLRequest.model_validate_json(request.model_dump_json()) == request
    True
"""

from pydantic import BaseModel, ConfigDict, Field

from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.foundation.random import MAX_SEED
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import ModelFamily


class SearchSettings(BaseModel):
    """O tamanho da procura: são os "hiperparâmetros" dos próprios agentes.

    Mais candidatos e mais folds dão uma escolha mais segura, e demoram mais. O formulário do
    ecrã Agentes gera-se deste esquema, como o dos hiperparâmetros de um modelo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    budget: int = Field(
        default=24,
        ge=1,
        le=200,
        title="Orçamento (candidatos)",
        description=(
            "Quantos candidatos medir, contando a triagem e a afinação. Cada candidato treina "
            "uma vez por fold: é isto que decide quanto tempo a procura demora."
        ),
    )
    folds: int = Field(
        default=5,
        ge=2,
        le=10,
        title="Folds da validação cruzada",
        description=(
            "Em quantas partes se divide o treino para medir cada candidato. Mais folds, "
            "estimativa mais segura e procura mais lenta."
        ),
    )
    finalists: int = Field(
        default=3,
        ge=1,
        le=10,
        title="Finalistas",
        description="Quantos modelos da triagem passam à afinação e repartem o resto do orçamento.",
    )
    max_rows: int = Field(
        default=2_000,
        ge=100,
        le=100_000,
        title="Linhas na procura",
        description=(
            "Com mais linhas de treino do que isto, a validação cruzada usa uma amostra, para a "
            "procura ser rápida. O vencedor treina no fim com o treino todo."
        ),
        json_schema_extra={"x-scale": "log"},
    )


class AutoMLRequest(BaseModel):
    """O pedido aos agentes: que modelo prevê melhor ``target`` nestes dados?

    Attributes:
        dataset: De onde vêm os dados.
        target: A coluna a prever.
        task_type: A tarefa; ``None`` aceita a sugestão do agente de dados.
        objective: A métrica a otimizar (o nome no registo, como ``"f1"``); ``None`` deixa o
            agente de dados sugerir.
        families: As famílias de modelos que participam; vazio = todas.
        include_didactic: Se entram também as implementações didáticas.
        preparation: Uma preparação feita à mão, que o agente de dados adota; ``None`` deixa-o
            decidir.
        settings: O tamanho da procura.
        seed: Seed raiz; ``None`` gera uma nova, que fica registada no resultado.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: SourceSpec
    target: str = Field(min_length=1)
    task_type: TaskType | None = None
    objective: str | None = None
    families: tuple[ModelFamily, ...] = ()
    include_didactic: bool = False
    preparation: PreparationConfig | None = None
    settings: SearchSettings = Field(default_factory=SearchSettings)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
