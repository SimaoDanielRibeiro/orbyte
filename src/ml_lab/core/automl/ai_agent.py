"""O agente de IA: propõe o próximo candidato da afinação a pensar, com um modelo de linguagem.

Os agentes de família sorteiam a próxima combinação de hiperparâmetros (procura aleatória): não
olham para o que já se mediu. O agente de IA olha. Na sua vez, recebe tudo o que a procura sabe
até ali (os finalistas, o espaço de procura de cada um e os resultados de todos os candidatos)
e responde com o próximo candidato: um dos finalistas e os valores dos hiperparâmetros, com a
razão numa frase.

O que o torna um agente de IA, e não de regras, é o ``LanguageModel``: a decisão vem de um
modelo de linguagem, que pensa antes de responder (o raciocínio chega aos bocados, e a
interface mostra-o ao vivo). O que o mantém honesto são as regras à volta dele (ADR 0042):

1. **a resposta tem um formato fixo** (um JSON Schema): o modelo não pode responder outra coisa;
2. **a proposta valida-se** contra o espaço de procura do modelo escolhido e contra o que já se
   experimentou. Uma proposta que não serve é recusada, com a razão, e não se mede;
3. **quem mede é o coordenador**, com validação cruzada só no treino. O agente de IA nunca vê o
   teste nem escolhe o vencedor.

Cada resposta fica guardada (``replies``): com elas, a procura repete-se sem o modelo de
linguagem, porque o ``ScriptedModel`` as dá outra vez, pela ordem.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ml_lab.core.automl.model_agent import ModelAgent
from ml_lab.core.automl.objective import Objective
from ml_lab.core.automl.search_log import AI_AGENT, Candidate
from ml_lab.core.automl.search_space import Dimension
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.llm import LanguageModel, Message, Reply, collect
from ml_lab.core.models.base import LabModel

MAX_REFUSALS: Final = 3
"""Quantas propostas recusadas seguidas fazem o agente de IA sair da afinação."""

type Finalist = tuple[ModelAgent, type[LabModel[Any]]]
"""Um modelo finalista e o agente da família dele (que sabe o espaço e o que já se experimentou)."""

INSTRUCTIONS: Final = (
    "És o agente de IA de uma procura de modelos de machine learning, num laboratório para "
    "aprender. Na tua vez, escolhes o próximo candidato a medir: um dos modelos finalistas e os "
    "valores dos hiperparâmetros que queres mudar, sempre dentro do espaço de procura de cada "
    "um. Olha para os resultados que já há: o que melhorou, o que piorou, e a distância entre o "
    "treino e a validação (uma distância grande quer dizer que o modelo está a decorar o "
    "treino). Não repitas uma combinação já medida. Não vês o teste e não escolhes o vencedor: "
    "o coordenador mede a tua proposta com validação cruzada, como as outras. Pensa em "
    "português de Portugal. Responde só com o JSON pedido; a razão vai em «reason», numa ou "
    "duas frases, em português de Portugal."
)
"""As instruções do agente de IA (a mensagem de sistema): o papel, as regras e o formato."""


@dataclass(frozen=True, slots=True)
class AiProposal:
    """Um candidato que o agente de IA propõe, já validado.

    Attributes:
        agent: O agente da família do modelo (é ele que guarda o que já se experimentou).
        model: O modelo.
        values: Os hiperparâmetros completos: os valores por omissão, com os propostos por cima.
        reason: Porquê, nas palavras do agente de IA.
    """

    agent: ModelAgent
    model: type[LabModel[Any]]
    values: dict[str, Any]
    reason: str


@dataclass(frozen=True, slots=True)
class AiTurn:
    """Uma vez do agente de IA: o que pensou e respondeu, e a proposta ou porque foi recusada.

    Attributes:
        reply: A resposta do modelo de linguagem, com o raciocínio.
        proposal: A proposta, se servir.
        problem: Se não servir, porquê.
    """

    reply: Reply
    proposal: AiProposal | None = None
    problem: str | None = None


class AiAgent:
    """O agente de IA da afinação.

    Args:
        llm: O modelo de linguagem (o Ollama, ou o ``ScriptedModel`` nos testes e ao repetir).
        objective: A métrica a otimizar.
        task_type: A tarefa.
        seed: A seed das respostas (cada vez usa a seguinte, para não responder sempre igual).
    """

    def __init__(
        self, llm: LanguageModel, objective: Objective, task_type: TaskType, *, seed: int
    ) -> None:
        self._llm = llm
        self._objective = objective
        self._task_type = task_type
        self._seed = seed
        self.replies: list[Reply] = []

    @property
    def name(self) -> str:
        """O nome a mostrar ("Agente · IA")."""
        return AI_AGENT

    @property
    def model_name(self) -> str:
        """O modelo de linguagem que pensa por ele ("qwen3:8b")."""
        return self._llm.name

    def propose(
        self,
        finalists: Sequence[Finalist],
        candidates: Sequence[Candidate],
        *,
        on_thinking: Callable[[str], None] | None = None,
    ) -> AiTurn:
        """Pensa no próximo candidato e propõe-no.

        Args:
            finalists: Os modelos que se podem escolher.
            candidates: Todos os candidatos medidos até aqui (a triagem e a afinação).
            on_thinking: Chamada com cada bocado do raciocínio, assim que chega.

        Raises:
            LanguageModelError: Se o modelo de linguagem não responder.
        """
        messages = [
            Message("system", INSTRUCTIONS),
            Message("user", briefing(self._objective, self._task_type, finalists, candidates)),
        ]
        reply = collect(
            self._llm.chat(
                messages,
                think=True,
                schema=answer_schema(finalists),
                seed=self._seed + len(self.replies),
            ),
            on_chunk=None if on_thinking is None else lambda chunk: on_thinking(chunk.thinking),
        )
        self.replies.append(reply)
        proposal, problem = read_answer(reply.content, finalists)
        return AiTurn(reply, proposal, problem)


def briefing(
    objective: Objective,
    task_type: TaskType,
    finalists: Sequence[Finalist],
    candidates: Sequence[Candidate],
) -> str:
    """O que o agente de IA sabe na sua vez: a tarefa, os finalistas e os resultados até aqui.

    Os números vão como em JSON (com ponto), porque é assim que o modelo os tem de responder.
    """
    lines = [
        f"Tarefa: {task_type.label.lower()}. Métrica a otimizar: {objective.label} "
        f"({objective.direction}), medida com validação cruzada só no treino.",
        "",
        "Os finalistas, e o que se pode mudar em cada um:",
    ]
    for agent, model in finalists:
        defaults = agent.defaults(model)
        lines.append(f"- {model.info.name} ({model.info.display_name}):")
        for dimension in agent.space(model).dimensions:
            default = json.dumps(defaults.get(dimension.name))
            lines.append(f"    - {dimension.name}: {_domain(dimension)}; por omissão {default}")
    lines += ["", "Os candidatos já medidos, pela ordem:"]
    lines += [_measured(candidate) for candidate in candidates] or ["(nenhum)"]
    lines += [
        "",
        "Propõe o próximo candidato: um dos finalistas, com valores dentro do espaço e uma "
        "combinação que ainda não se mediu. Em «values», põe só os hiperparâmetros que mudas.",
    ]
    return "\n".join(lines)


def answer_schema(finalists: Sequence[Finalist]) -> dict[str, Any]:
    """O formato da resposta: o modelo (um dos finalistas), os valores e a razão.

    O Ollama obriga o modelo a responder neste formato (saída estruturada). Um hiperparâmetro
    com o mesmo nome em dois finalistas aceita o espaço de qualquer um deles: o que conta é o
    do modelo escolhido, e isso valida-se depois (``read_answer``).
    """
    shapes: dict[str, list[dict[str, Any]]] = {}
    for agent, model in finalists:
        for dimension in agent.space(model).dimensions:
            shape = dimension.schema()
            if shape not in shapes.setdefault(dimension.name, []):
                shapes[dimension.name].append(shape)
    properties = {
        name: options[0] if len(options) == 1 else {"anyOf": options}
        for name, options in shapes.items()
    }
    return {
        "type": "object",
        "properties": {
            "model": {"type": "string", "enum": [model.info.name for _, model in finalists]},
            "values": {"type": "object", "properties": properties},
            "reason": {"type": "string"},
        },
        "required": ["model", "values", "reason"],
    }


def read_answer(
    content: str, finalists: Sequence[Finalist]
) -> tuple[AiProposal | None, str | None]:
    """A proposta escrita na resposta, se servir; se não, porquê.

    Serve se o modelo for um dos finalistas, se os valores estiverem no espaço de procura dele
    e se a combinação ainda não tiver sido medida. Os hiperparâmetros que a resposta não traz
    ficam com o valor por omissão.
    """
    try:
        answer = json.loads(content)
    except ValueError:
        return None, "a resposta não é o JSON pedido"
    if not isinstance(answer, dict):
        return None, "a resposta não é um objeto JSON"
    chosen = answer.get("model")
    entry = next((entry for entry in finalists if entry[1].info.name == chosen), None)
    if entry is None:
        names = ", ".join(model.info.name for _, model in finalists)
        return None, f"«{chosen}» não é um dos finalistas ({names})"
    agent, model = entry
    raw = answer.get("values") or {}
    if not isinstance(raw, dict):
        return None, "os valores têm de vir num objeto, com o nome de cada hiperparâmetro"
    space = agent.space(model)
    values = space.normalized(raw)
    problems = space.problems(values)
    if problems:
        return None, "; ".join(problems)
    full = {**agent.defaults(model), **values}
    if agent.has_tried(model, full):
        return None, f"essa combinação de {model.info.display_name} já foi medida"
    reason = str(answer.get("reason") or "").strip() or "Não disse porquê."
    return AiProposal(agent, model, full, reason), None


def _domain(dimension: Dimension) -> str:
    """O espaço de uma dimensão, para o modelo de linguagem ler."""
    if dimension.values:
        return f"um destes: {json.dumps(list(dimension.values))}"
    low, high = dimension.bounds
    if dimension.integer:
        return f"inteiro entre {int(low)} e {int(high)}"
    scale = " (escala logarítmica)" if dimension.log else ""
    return f"número entre {low:g} e {high:g}{scale}"


def _measured(candidate: Candidate) -> str:
    """Um candidato medido, numa linha: o modelo, o que mudou e o resultado."""
    changes = json.dumps(candidate.changes) if candidate.changes else "valores por omissão"
    head = f"#{candidate.number} {candidate.model} {changes}"
    if candidate.failed or candidate.mean is None:
        return f"{head}: falhou ({candidate.error})"
    train = "" if candidate.train_mean is None else f", no treino {candidate.train_mean:.4f}"
    return f"{head}: {candidate.mean:.4f} ± {(candidate.std or 0.0):.4f}{train}"
