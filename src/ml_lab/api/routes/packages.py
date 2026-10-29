"""Rotas dos pacotes dos modelos (ecrã Guardar): guardar, a biblioteca e carregar (ADR 0039).

Um pacote (``.mllab``) leva a preparação aprendida, o modelo e a ficha dele. Aqui vê-se o
pacote de um run antes de existir, grava-se, descarrega-se, e carrega-se um pacote feito
noutro sítio. O descarregar é a única rota que não devolve JSON: devolve o próprio ficheiro.
"""

from typing import Final

from fastapi import APIRouter, UploadFile
from fastapi.responses import Response

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.core.schemas.packages import PackageRequest
from ml_lab.core.services import (
    PackageCandidate,
    PackageOptions,
    PackagePreview,
    PackageView,
    SavedPackage,
)

router = ServiceRouter(prefix="/packages", tags=["packages"], responses=ERROR_RESPONSES)


@router.get("/candidates")
def candidates(lab: LabDep) -> tuple[PackageCandidate, ...]:
    """Os runs do histórico que se podem guardar num pacote (ou porque não se podem)."""
    return lab.packages.candidates()


@router.get("/runs/{run_id}/options")
def options(lab: LabDep, run_id: str) -> PackageOptions:
    """O nome e as partes sugeridas para o pacote de um run."""
    return lab.packages.options(run_id)


@router.post("/runs/{run_id}/preview")
def preview(lab: LabDep, run_id: str, request: PackageRequest) -> PackagePreview:
    """O pacote de um run, parte a parte, antes de existir (é o que o guardar grava)."""
    return lab.packages.preview(run_id, request)


@router.post("/runs/{run_id}", status_code=201)
def save(lab: LabDep, run_id: str, request: PackageRequest) -> SavedPackage:
    """Grava o pacote de um run na biblioteca e guarda o run no histórico."""
    return lab.packages.save(run_id, request)


@router.get("")
def saved(lab: LabDep) -> tuple[SavedPackage, ...]:
    """Os pacotes da biblioteca, do mais recente para o mais antigo."""
    return lab.packages.saved()


@router.post("/uploads", status_code=201)
def load(lab: LabDep, file: UploadFile) -> SavedPackage:
    """Carrega um ficheiro .mllab: entra na biblioteca e, se não houver risco, no histórico.

    Um pacote feito neste ML Lab (ou sem o modelo) entra logo no histórico. Um de outra
    instalação fica só para ler até se confirmar que se confia nele (``trust``).
    """
    return lab.packages.load(file.filename or "modelo.mllab", file.file.read())


@router.get("/{package_id}")
def contents(lab: LabDep, package_id: str) -> PackageView:
    """Um pacote da biblioteca, aberto parte a parte."""
    return lab.packages.contents(package_id)


@router.post("/{package_id}/trust")
def trust(lab: LabDep, package_id: str) -> SavedPackage:
    """Abre o modelo de um pacote de outra instalação, depois do aviso, e põe o run no histórico."""
    return lab.packages.trust(package_id)


@router.delete("/{package_id}", status_code=204)
def delete(lab: LabDep, package_id: str) -> None:
    """Apaga um pacote da biblioteca (o run que ele pôs no histórico fica lá)."""
    lab.packages.delete(package_id)


MEDIA_TYPE: Final = "application/zip"
"""O tipo de um ``.mllab``: é um ZIP."""

files = APIRouter(prefix="/packages", tags=["packages"], responses=ERROR_RESPONSES)
"""As rotas que devolvem ficheiros, e não JSON (por isso não passam pelo ``ServiceRouter``)."""


@files.get(
    "/{package_id}/file",
    response_class=Response,
    summary="Descarrega o ficheiro .mllab de um pacote da biblioteca",
    responses={200: {"content": {MEDIA_TYPE: {}}, "description": "O ficheiro .mllab."}},
)
def download(lab: LabDep, package_id: str) -> Response:
    """O ficheiro vem com o nome do pacote, pronto a guardar no disco."""
    name, data = lab.packages.file(package_id)
    return Response(
        content=data,
        media_type=MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
