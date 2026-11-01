"""As rotas da API, um módulo por serviço do ``Lab``.

Cada rota tem o nome do método do serviço que chama e só o chama: a lógica fica toda nos
serviços (regra 1 do ``CONTRIBUTING.md``). Uma rota nova entra na lista ``ROUTERS``.
"""

from fastapi import APIRouter

from ml_lab.api.routes import (
    analysis,
    automl,
    catalog,
    datasets,
    experiments,
    explanations,
    history,
    laboratory,
    llm,
    meta,
    packages,
    reinforcement,
)

ROUTERS: tuple[APIRouter, ...] = (
    datasets.router,
    catalog.router,
    experiments.router,
    analysis.router,
    history.router,
    laboratory.router,
    reinforcement.router,
    automl.router,
    packages.router,
    packages.files,
    llm.router,
    explanations.router,
    meta.router,
)
