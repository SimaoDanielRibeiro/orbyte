"""Ligação das rotas aos serviços: o ``Lab`` de cada pedido e as respostas em JSON.

Cada rota chama um método de um serviço e devolve o que ele devolve, sem mais lógica. Três
regras valem para todas as rotas e por isso ficam aqui, escritas uma vez:

- **as respostas passam a JSON pelo** ``to_json`` (ADR 0019), o único sítio do projeto que
  o faz: os NaN passam a ``null`` e os valores do NumPy a números, tal como no registo de
  experiências. O FastAPI continua a descrever cada resposta pelo tipo que a rota anuncia, e
  é daí que vem o OpenAPI de ``/docs``;
- **o identificador de cada operação é** ``<serviço>_<método>`` (por exemplo
  ``datasets_samples``). É o nome que o cliente gerado para o frontend vai ter, e é com ele
  que um teste confirma que cada método público dos serviços tem a sua rota;
- **o resumo de cada rota em** ``/docs`` **é a primeira linha da sua docstring**, em
  português, e o resto da docstring é a descrição.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.requests import HTTPConnection

from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.services import Lab


def get_lab(connection: HTTPConnection) -> Lab:
    """Os serviços da aplicação: um ``Lab`` por processo, criado com a aplicação."""
    lab: Lab = connection.app.state.lab
    return lab


LabDep = Annotated[Lab, Depends(get_lab)]
"""O ``Lab``, pedido como parâmetro de uma rota (``lab: LabDep``)."""


def operation_id(route: APIRoute) -> str:
    """O identificador da operação no OpenAPI: o serviço (a etiqueta) e o nome da rota."""
    return f"{route.tags[0]}_{route.name}"


class ServiceRouter(APIRouter):
    """Um ``APIRouter`` cujas rotas respondem com o resultado do serviço, passado pelo ``to_json``.

    Também tira o resumo e a descrição de cada rota da sua docstring.
    """

    def add_api_route(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        """Regista a rota com o ``endpoint`` embrulhado (ver o módulo)."""
        summary, _, description = inspect.cleandoc(endpoint.__doc__ or "").partition("\n")
        # Os decoradores (@router.get...) passam summary=None quando não se dá nenhum.
        kwargs["summary"] = kwargs.get("summary") or summary or None
        kwargs["description"] = kwargs.get("description") or description.strip() or None
        wrapped = _json_endpoint(endpoint, kwargs.get("status_code"))
        # Sem descrição, o FastAPI usa a docstring inteira, e o resumo aparecia duas vezes em
        # /docs: na rota embrulhada, a docstring fica só com o resto.
        wrapped.__doc__ = description.strip()
        super().add_api_route(path, wrapped, **kwargs)


def _json_endpoint(endpoint: Callable[..., Any], status_code: int | None) -> Callable[..., Any]:
    """O ``endpoint`` a devolver uma resposta JSON feita pelo ``to_json``.

    O ``functools.wraps`` guarda a assinatura original: o FastAPI continua a ver os
    parâmetros (de onde lê o pedido) e o tipo devolvido (de onde descreve a resposta).
    """
    status = status_code or 200
    if inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        async def run_async(*args: Any, **kwargs: Any) -> Response:
            return _response(await endpoint(*args, **kwargs), status)

        return run_async

    @functools.wraps(endpoint)
    def run(*args: Any, **kwargs: Any) -> Response:
        return _response(endpoint(*args, **kwargs), status)

    return run


def _response(value: Any, status: int) -> Response:
    if isinstance(value, Response):
        return value
    if value is None and status == 204:  # apagar: sem corpo
        return Response(status_code=status)
    return JSONResponse(to_json(value), status_code=status)
