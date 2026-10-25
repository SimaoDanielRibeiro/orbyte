"""A aplicação FastAPI do ML Lab: os serviços por HTTP, com a documentação em ``/docs``.

Abre-se com ``uv run python -m ml_lab.api`` e fica em http://127.0.0.1:8000 (a
documentação interativa em ``/docs``). A API é uma camada fina por cima dos serviços (ADR
0022): cada rota chama um método de um serviço, e é tudo.

Há um só ``Lab`` por processo, criado com a aplicação: o ML Lab é local e de um utilizador,
por isso os datasets carregados e os treinos lançados são os mesmos para todos os pedidos.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_lab import __version__
from ml_lab.api.errors import add_error_handlers
from ml_lab.api.routes import ROUTERS
from ml_lab.api.routing import operation_id
from ml_lab.config import Settings, get_settings
from ml_lab.core.services import Lab
from ml_lab.logging import configure_logging
from ml_lab.process import run_at_full_speed

DESCRIPTION = """
Os serviços do ML Lab por HTTP: carregar e preparar dados, treinar em segundo plano,
acompanhar o treino, avaliar, explicar e comparar runs.

**Um percurso para começar, aqui mesmo:**

1. `POST /datasets/uploads` envia um CSV (ou `POST /datasets/samples/iris` carrega um
   exemplo). A resposta tem o `id` do dataset e a origem (`source`).
2. `POST /experiments` começa um treino com a origem, a preparação e o modelo (por exemplo
   `{"dataset": <source>, "preparation": {"target": "species"}, "model": {"name":
   "logistic_regression_gd"}, "seed": 42}`). A resposta tem o `run_id`.
3. `GET /experiments/{run_id}/updates` mostra o progresso, os passos e a fronteira do
   último snapshot (o WebSocket `/experiments/{run_id}/stream` envia o mesmo sozinho).
4. `GET /experiments/{run_id}/result` dá o resultado quando o treino termina.

**O modo automático:** `POST /automl` com `{"dataset": <source>, "target": "Survived"}` põe
os agentes a procurar o melhor modelo. `GET /automl/{search_id}/updates` traz a conversa
dos agentes e os candidatos novos, e `GET /automl/{search_id}/result` o quadro final, com o
`run_id` do vencedor.

Os erros vêm sempre com um `code` estável e uma `message` em português que diz o que fazer.
"""

TAGS = [
    {"name": "datasets", "description": "Carregar, explorar e preparar dados."},
    {"name": "catalog", "description": "O catálogo de modelos, com as fichas e os formulários."},
    {
        "name": "experiments",
        "description": "Treinar em segundo plano, acompanhar o treino e pedir o resultado.",
    },
    {
        "name": "analysis",
        "description": "Depois do treino: gráficos, validação cruzada e explicações.",
    },
    {
        "name": "history",
        "description": "O histórico de runs: guardar, abrir, comparar e reproduzir.",
    },
    {"name": "laboratory", "description": "As demonstrações do ecrã Laboratório."},
    {
        "name": "reinforcement",
        "description": "Treinar agentes de reinforcement learning num ambiente, e acompanhá-los.",
    },
    {
        "name": "automl",
        "description": (
            "O modo automático: os agentes preparam os dados, experimentam os modelos e "
            "escolhem o melhor para o pedido, com a razão de cada decisão."
        ),
    },
    {"name": "explanations", "description": "Os textos do painel «Porquê?»."},
    {"name": "meta", "description": "Sobre a própria API."},
]


def create_app(lab: Lab | None = None, *, settings: Settings | None = None) -> FastAPI:
    """Cria a aplicação.

    Args:
        lab: Os serviços; por omissão, os do processo (com o registo em SQLite). Os testes
            passam os seus.
        settings: As definições (o CORS e, sem ``lab``, as pastas e os logs).
    """
    settings = settings or get_settings()
    if lab is None:
        configure_logging(settings.log_level, settings.log_format)
        settings.ensure_directories()
        run_at_full_speed()  # o servidor corre num terminal: o Windows travava-o (ml_lab.process)
        lab = Lab.create(settings)
    app = FastAPI(
        title="ML Lab",
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        generate_unique_id_function=operation_id,
    )
    app.state.lab = lab
    # O navegador só deixa uma página chamar a API se a API disser que essa origem pode.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    add_error_handlers(app)
    for router in ROUTERS:
        app.include_router(router)
    return app
