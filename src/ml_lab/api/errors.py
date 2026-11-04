"""Os erros da API: cada erro do ML Lab vira um código HTTP, com a mensagem em inglês.

Os serviços lançam subclasses de ``MLLabError`` com mensagens que dizem o que aconteceu e o
que fazer (regra 8 do ``CONTRIBUTING.md``). A API não reescreve as mensagens: só escolhe o código
HTTP pela categoria do erro, a primeira desta lista que servir:

1. ``TrainingStateError`` → **409**: o pedido não combina com o estado do treino (ainda
   decorre, ou falhou);
2. ``DataError`` e ``ValueError`` → **422**: os valores do pedido não servem (um
   hiperparâmetro fora dos limites, uma coluna que não existe nos dados);
3. ``LookupError`` → **404**: o identificador não existe (um run, um dataset, um modelo);
4. ``TrackingError`` → **500**: o registo de experiências falhou do lado do servidor;
5. outro ``MLLabError`` → **400**.

O corpo da resposta é sempre um ``ApiError``, com um ``code`` estável (o nome do erro, em
``snake_case``: ``run_not_found``) com que o frontend decide o que fazer. Um pedido com a
forma errada (um campo em falta, texto onde devia vir um número) responde 422 com o código
``invalid_request`` e uma frase em inglês por cada problema.
"""

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.schemas.validation import describe_problems
from ml_lab.core.services import DataError, MLLabError, TrackingError, TrainingStateError
from ml_lab.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ApiError:
    """Um erro, como a API o devolve.

    Attributes:
        code: Identificador estável do erro (``run_not_found``, ``invalid_request``...).
        message: O que aconteceu e o que fazer, em inglês.
        problems: Num pedido inválido, uma frase por cada problema.
    """

    code: str
    message: str
    problems: tuple[str, ...] = ()


STATUS_BY_CATEGORY: Final[tuple[tuple[type[Exception], int], ...]] = (
    (TrainingStateError, 409),
    (DataError, 422),
    (ValueError, 422),
    (LookupError, 404),
    (TrackingError, 500),
    (MLLabError, 400),
)
"""O código HTTP de cada categoria de erro, pela ordem em que se procuram (ver o módulo)."""

ERROR_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    400: {"model": ApiError, "description": "O pedido não se pode cumprir."},
    404: {"model": ApiError, "description": "O identificador não existe."},
    409: {"model": ApiError, "description": "O treino ainda decorre, ou falhou."},
    422: {"model": ApiError, "description": "Os valores do pedido não servem."},
    500: {
        "model": ApiError,
        "description": "O registo de experiências falhou, ou um erro inesperado.",
    },
}
"""As respostas de erro, para o OpenAPI as descrever em cada rota (todas com um ``ApiError``)."""

_SOURCES: Final = frozenset({"body", "query", "path", "header", "cookie"})


def status_for(error: Exception) -> int:
    """O código HTTP de um erro (500 para os que não são do ML Lab)."""
    return next((status for kind, status in STATUS_BY_CATEGORY if isinstance(error, kind)), 500)


def error_code(error: Exception) -> str:
    """O nome do erro em ``snake_case``, sem o ``Error`` (``RunNotFoundError``: ``run_not_found``).

    É o ``code`` que o frontend usa para decidir o que fazer: estável, porque os nomes dos
    erros não mudam.
    """
    name = type(error).__name__.removesuffix("Error")
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def api_error(error: Exception) -> ApiError:
    """O ``ApiError`` de um erro do ML Lab."""
    return ApiError(code=error_code(error), message=str(error))


def add_error_handlers(app: FastAPI) -> None:
    """Liga os erros às respostas descritas no módulo."""
    app.add_exception_handler(MLLabError, _ml_lab_error)
    app.add_exception_handler(RequestValidationError, _invalid_request)
    app.add_exception_handler(Exception, _unexpected)


async def _ml_lab_error(request: Request, error: Exception) -> JSONResponse:
    status = status_for(error)
    if status >= 500:
        logger.error("erro do servidor", extra={"path": request.url.path, "error": str(error)})
    return JSONResponse(to_json(api_error(error)), status_code=status)


async def _invalid_request(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)  # só este handler o recebe
    problems = describe_problems(_without_source(error.errors()))
    body = ApiError(
        code="invalid_request",
        message="The request has values that don't work: " + " ".join(problems),
        problems=problems,
    )
    return JSONResponse(to_json(body), status_code=422)


async def _unexpected(request: Request, error: Exception) -> JSONResponse:
    logger.error("erro inesperado", extra={"path": request.url.path}, exc_info=error)
    body = ApiError(
        code="unexpected",
        message=(
            f"Unexpected server error ({type(error).__name__}): {error}. The server logs have "
            "the details."
        ),
    )
    return JSONResponse(to_json(body), status_code=500)


def _without_source(problems: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Os problemas sem a parte do pedido no caminho (``body.model.name`` → ``model.name``)."""
    for problem in problems:
        location = tuple(problem.get("loc", ()))
        if location and location[0] in _SOURCES:
            location = location[1:]
        yield {**problem, "loc": location}
