"""Rotas dos agentes (ecrã Agentes): o modo automático, que procura o melhor modelo para um pedido.

Funciona como o treino: ``POST /automl`` valida o pedido, começa a procura em segundo plano
e responde logo (202) com o estado. Acompanha-se com ``GET .../updates`` ou pelo WebSocket
``.../stream`` (os candidatos e as mensagens novas da conversa dos agentes), e o resultado
pede-se com ``GET .../result``. Cada candidato medido é um passo.

``POST .../candidates/{number}/train`` é a ponte para o modo manual: treina um candidato
como um run normal, que se acompanha nas rotas de ``/experiments``.

Depois de a procura acabar, o analista (fase 13, ADR 0053) responde a perguntas sobre ela:
``POST .../questions`` faz a pergunta e responde logo (202) com o estado; a pergunta acompanha-se
pelo seu id em ``/automl/questions/{question_id}/...`` (``updates``, o WebSocket ``stream``,
``result`` e ``cancel``), como uma procura.
"""

from typing import Annotated

from fastapi import Query, WebSocket

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.api.stream import stream_updates
from ml_lab.core.services import (
    AnalystAnswer,
    AnalystOptions,
    AnalystRequest,
    AnalystUpdate,
    AutoMLOptions,
    AutoMLPlan,
    AutoMLRequest,
    AutoMLResult,
    AutoMLUpdate,
    JobState,
    TaskType,
)

router = ServiceRouter(prefix="/automl", tags=["automl"], responses=ERROR_RESPONSES)


@router.get("/options")
def options(
    lab: LabDep,
    task_type: TaskType,
    objective: Annotated[
        str | None, Query(description="A métrica a otimizar; por omissão, a principal.")
    ] = None,
) -> AutoMLOptions:
    """O que se pode pedir aos agentes numa tarefa: as métricas, os agentes e o orçamento."""
    return lab.automl.options(task_type, objective)


@router.post("/plan")
def plan(lab: LabDep, request: AutoMLRequest) -> AutoMLPlan:
    """O que os agentes vão fazer com o pedido, sem começar: as decisões do agente de dados."""
    return lab.automl.plan(request)


@router.post("", status_code=202)
def start(lab: LabDep, request: AutoMLRequest) -> JobState:
    """Valida o pedido e procura em segundo plano (cada candidato medido é um passo)."""
    return lab.automl.start(request).state()


@router.get("/{search_id}/job")
def job(lab: LabDep, search_id: str) -> JobState:
    """O estado da procura: em curso, em pausa, terminada ou falhada, e o progresso."""
    return lab.automl.job(search_id)


@router.post("/{search_id}/pause")
def pause(lab: LabDep, search_id: str) -> JobState:
    """Pede uma pausa no fim do candidato atual."""
    return lab.automl.pause(search_id)


@router.post("/{search_id}/resume")
def resume(lab: LabDep, search_id: str) -> JobState:
    """Retoma uma procura em pausa."""
    return lab.automl.resume(search_id)


@router.post("/{search_id}/cancel")
def cancel(lab: LabDep, search_id: str) -> JobState:
    """Pede à procura que pare no fim do candidato atual (o vencedor é o melhor até aí)."""
    return lab.automl.cancel(search_id)


@router.get("/{search_id}/updates")
def updates(
    lab: LabDep,
    search_id: str,
    after_step: Annotated[
        int, Query(ge=0, description="O número do último candidato que já tens.")
    ] = 0,
    after_message: Annotated[
        int, Query(ge=0, description="O número da última mensagem que já tens.")
    ] = 0,
) -> AutoMLUpdate:
    """O que mudou na procura: o estado, os candidatos novos e as mensagens novas."""
    return lab.automl.updates(search_id, after_step, after_message)


@router.get("/{search_id}/plan")
def search_plan(lab: LabDep, search_id: str) -> AutoMLPlan:
    """O plano de uma procura (as decisões do agente de dados e os agentes), mesmo a decorrer."""
    return lab.automl.search_plan(search_id)


@router.get("/{search_id}/result")
def result(lab: LabDep, search_id: str) -> AutoMLResult:
    """O resultado de uma procura terminada: o quadro, a conversa e o run do vencedor."""
    return lab.automl.result(search_id)


@router.post("/{search_id}/candidates/{number}/train", status_code=202)
def train_candidate(lab: LabDep, search_id: str, number: int) -> JobState:
    """Treina um candidato como um run normal (o modo manual a partir do que os agentes acharam)."""
    return lab.automl.train_candidate(search_id, number).state()


@router.websocket("/{search_id}/stream")
async def stream(websocket: WebSocket, lab: LabDep, search_id: str) -> None:
    """Envia as mudanças da procura (o mesmo que ``updates``) até ela terminar."""

    def poll(after_step: int, after_message: int) -> AutoMLUpdate:
        return lab.automl.updates(search_id, after_step, after_message)

    await stream_updates(websocket, poll)


# --- o analista (fase 13, ADR 0053) ----------------------------------------------------


@router.get("/{search_id}/analyst")
def analyst_options(lab: LabDep, search_id: str) -> AnalystOptions:
    """O que se pode perguntar ao analista sobre uma procura: as sugestões e as ferramentas."""
    return lab.automl.analyst_options(search_id)


@router.get("/{search_id}/questions")
def questions(lab: LabDep, search_id: str) -> tuple[AnalystAnswer, ...]:
    """As perguntas já respondidas sobre uma procura, pela ordem em que se fizeram."""
    return lab.automl.questions(search_id)


@router.post("/{search_id}/questions", status_code=202)
def ask(lab: LabDep, search_id: str, request: AnalystRequest) -> JobState:
    """Pergunta ao analista sobre uma procura acabada; ele responde em segundo plano.

    Sem respostas guardadas (``replies``), é preciso o Ollama: sem ele, responde 400 com o que
    fazer (o estado está em ``GET /llm/status``).
    """
    return lab.automl.ask(search_id, request).state()


@router.get("/questions/{question_id}/job")
def question_job(lab: LabDep, question_id: str) -> JobState:
    """O estado de uma pergunta ao analista: em curso, terminada ou falhada."""
    return lab.automl.question_job(question_id)


@router.get("/questions/{question_id}/updates")
def question_updates(
    lab: LabDep,
    question_id: str,
    after_step: Annotated[
        int, Query(ge=0, description="O número do último passo do analista que já tens.")
    ] = 0,
) -> AnalystUpdate:
    """O que mudou na pergunta: os passos novos do analista e o que está a pensar agora."""
    return lab.automl.question_updates(question_id, after_step)


@router.get("/questions/{question_id}/result")
def question_result(lab: LabDep, question_id: str) -> AnalystAnswer:
    """A resposta do analista, com o raciocínio, cada ferramenta usada e o que devolveu."""
    return lab.automl.question_result(question_id)


@router.post("/questions/{question_id}/cancel")
def cancel_question(lab: LabDep, question_id: str) -> JobState:
    """Pede ao analista que pare, mesmo a meio de uma resposta do modelo de linguagem."""
    return lab.automl.cancel_question(question_id)


@router.websocket("/questions/{question_id}/stream")
async def question_stream(websocket: WebSocket, lab: LabDep, question_id: str) -> None:
    """Envia as mudanças da pergunta (o mesmo que ``question_updates``) até ela terminar."""

    def poll(after_step: int, _after_snapshot: int) -> AnalystUpdate:
        return lab.automl.question_updates(question_id, after_step)

    await stream_updates(websocket, poll)
