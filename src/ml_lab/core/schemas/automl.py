"""O pedido aos agentes (fase 9): os dados, o alvo, a métrica e o tamanho da procura (Pydantic).

É o que entra no serviço dos agentes e o que fica no resultado, com a seed já resolvida: com
o mesmo pedido, a procura repete-se tal e qual. Vai e volta de JSON sem perder nada, por isso
serve também a API. A pergunta ao analista (``AnalystRequest``, fase 13) segue a mesma ideia: a
resposta traz o pedido com as respostas do modelo de linguagem, e esse pedido repete-a.

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

from typing import Any

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
        title="Budget (candidates)",
        description=(
            "How many candidates to measure, counting the screening and the tuning. Each "
            "candidate trains once per fold: this is what decides how long the search takes."
        ),
    )
    folds: int = Field(
        default=5,
        ge=2,
        le=10,
        title="Cross-validation folds",
        description=(
            "How many parts the training data is split into to measure each candidate. More "
            "folds, a safer estimate and a slower search."
        ),
    )
    finalists: int = Field(
        default=3,
        ge=1,
        le=10,
        title="Finalists",
        description=(
            "How many models from the screening move on to the tuning and share the rest of the "
            "budget."
        ),
    )
    max_rows: int = Field(
        default=2_000,
        ge=100,
        le=100_000,
        title="Rows in the search",
        description=(
            "With more training rows than this, cross-validation uses a sample, so the search "
            "is fast. The winner trains at the end on all the training data."
        ),
        json_schema_extra={"x-scale": "log"},
    )


class RecordedToolCall(BaseModel):
    """Uma ferramenta que o modelo de linguagem pediu, guardada com a resposta (o analista).

    Attributes:
        name: O nome da ferramenta (``"list_candidates"``).
        arguments: Os argumentos, como o modelo os escreveu.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class RecordedReply(BaseModel):
    """Uma resposta de um agente de IA, guardada para se repetir sem o modelo de linguagem.

    Serve os dois agentes de IA (ADR 0042 e 0053): o da afinação responde com o JSON da proposta
    em ``content``; o analista pede ferramentas (``tool_calls``) e, no fim, responde por palavras.

    Attributes:
        thinking: O raciocínio.
        content: A resposta, como o modelo a escreveu.
        tool_calls: As ferramentas que o modelo pediu nesta resposta (só o analista as usa).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    thinking: str = ""
    content: str = ""
    tool_calls: tuple[RecordedToolCall, ...] = ()


class AnalystRequest(BaseModel):
    """Uma pergunta ao analista sobre uma procura acabada (fase 13, ADR 0053).

    Attributes:
        question: A pergunta, por palavras ("Why did the winner win?").
        replies: As respostas do modelo de linguagem de uma pergunta anterior: com elas, o
            analista responde outra vez tal e qual, sem o Ollama (o ``ScriptedModel`` dá-as pela
            ordem). A resposta traz sempre as que usou.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=500)
    replies: tuple[RecordedReply, ...] = ()


class AutoMLRequest(BaseModel):
    """O pedido aos agentes: que modelo prevê melhor ``target`` nestes dados?

    Attributes:
        dataset: De onde vêm os dados.
        target: A coluna a prever.
        task_type: A tarefa; ``None`` aceita a sugestão do agente de dados.
        objective: A métrica a otimizar (o nome no registo, como ``"f1"``); ``None`` deixa o
            agente de dados sugerir.
        families: As famílias de modelos que participam; vazio = todas.
        models: Os modelos que participam, pelo nome no registo; vazio = todos os que podem
            (ADR 0049). Com as famílias, entra o que estiver nas duas escolhas.
        preparation: Uma preparação feita à mão, que o agente de dados adota; ``None`` deixa-o
            decidir.
        settings: O tamanho da procura.
        seed: Seed raiz; ``None`` gera uma nova, que fica registada no resultado.
        ai_agent: Se o agente de IA entra na afinação (precisa do Ollama, ADR 0042).
        ai_replies: As respostas do agente de IA de uma procura anterior: com elas, a procura
            repete-se tal e qual, sem o Ollama. O resultado traz sempre as que usou.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: SourceSpec
    target: str = Field(min_length=1)
    task_type: TaskType | None = None
    objective: str | None = None
    families: tuple[ModelFamily, ...] = ()
    models: tuple[str, ...] = ()
    preparation: PreparationConfig | None = None
    settings: SearchSettings = Field(default_factory=SearchSettings)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    ai_agent: bool = False
    ai_replies: tuple[RecordedReply, ...] = ()
