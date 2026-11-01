"""Serviço dos pacotes: guardar o que se preparou e aprendeu num ficheiro, e voltar a abri-lo.

É o ecrã Guardar (fase 11, ADR 0039). Um pacote (``.mllab``) leva tudo o que é preciso para
perceber um modelo e para voltar a prever com ele, sem a fonte dos dados:

- **guardar:** ``candidates`` diz que runs se podem guardar; ``options`` sugere o nome e as
  partes opcionais; ``preview`` mostra o ficheiro antes de ele existir, parte a parte; e
  ``save`` grava-o na biblioteca e guarda o run no histórico;
- **a biblioteca:** ``saved``, ``contents`` (um pacote aberto no mesmo leitor da
  pré-visualização), ``delete`` e o ficheiro para descarregar;
- **carregar:** ``load`` põe um ``.mllab`` na biblioteca. Um pacote feito aqui (ou sem o
  modelo, que não tem pickle) entra logo no histórico; um de outra instalação espera que a
  pessoa confirme que confia nele (``trust``), porque abrir o modelo corre o código de quem o
  fez.

A pré-visualização e o ficheiro saem do mesmo código (``_assemble``): o que se vê é o que se
grava.
"""

import hashlib
import json
import platform
import re
import secrets
import threading
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError as DistributionNotFound
from importlib.metadata import version
from typing import Any, Final, Literal

import numpy as np

from ml_lab.config import Settings, get_settings
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.serving import FittedPreparation, lineage_document
from ml_lab.core.export import export_onnx, onnx_signature
from ml_lab.core.foundation.errors import (
    CapabilityError,
    PackageError,
    RunNotFoundError,
    RunNotInMemoryError,
)
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, ModelFamily, ModelInfo
from ml_lab.core.schemas.packages import PackageRequest
from ml_lab.core.schemas.serialization import from_json, internal, to_json
from ml_lab.core.services.experiments import ExperimentResult, ExperimentService, RunArtifacts
from ml_lab.core.services.history import HistoryService
from ml_lab.core.services.model_card import CardFile, model_card
from ml_lab.core.tracking import packages as files
from ml_lab.core.tracking.packages import (
    LibraryEntry,
    OpenedPackage,
    PackageContents,
    PackageFile,
    PackageKey,
    PackageLibrary,
)
from ml_lab.core.visualization_data.animation import TrainingAnimation
from ml_lab.core.visualization_data.network import NetworkTrace
from ml_lab.logging import get_logger

logger = get_logger(__name__)

PREVIEW_CHARACTERS: Final = 200_000
"""Quantos caracteres de um ficheiro de texto mostra a pré-visualização (o resto fica no ZIP)."""

LARGE_PACKAGE: Final = 50 * 1024 * 1024
"""A partir deste tamanho (sem comprimir), o pacote avisa que é grande."""

_EXAMPLE_SOURCES: Final = frozenset({"sample", "synthetic"})
"""As origens cujas linhas se podem mostrar à vontade (as dos ficheiros podem ser de alguém)."""

ONNX_CACHE: Final = 8
"""Quantos ``model.onnx`` (ou as razões por que não os há) ficam guardados, um por run."""

_VERSIONED: Final = (
    "ml-lab",
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "joblib",
    "onnx",
    "skl2onnx",
    "onnxruntime",
)
"""As bibliotecas cujas versões o manifesto regista."""

_PICKLED: Final = ("scikit-learn", "torch", "numpy")
"""As bibliotecas de que um pickle depende: uma versão diferente pede um aviso."""


class PackageFileKind(StrEnum):
    """O tipo de um ficheiro de um pacote (diz como se pré-visualiza)."""

    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    BINARY = "binary"

    @property
    def label(self) -> str:
        """Nome a mostrar."""
        return _KIND_LABELS[self]


_KIND_LABELS: Final = {
    PackageFileKind.MARKDOWN: "Markdown",
    PackageFileKind.JSON: "JSON",
    PackageFileKind.CSV: "CSV",
    PackageFileKind.TEXT: "Texto",
    PackageFileKind.BINARY: "Binário",
}


class PackageOrigin(StrEnum):
    """De onde veio um pacote da biblioteca."""

    MADE_HERE = "made_here"
    LOADED = "loaded"

    @property
    def label(self) -> str:
        """Nome a mostrar."""
        return "Feito aqui" if self is PackageOrigin.MADE_HERE else "Carregado"


@dataclass(frozen=True, slots=True)
class _PartInfo:
    title: str
    description: str
    kind: PackageFileKind


_PARTS: Final[Mapping[str, _PartInfo]] = {
    files.README: _PartInfo(
        "A ficha do modelo",
        "O que prevê, com que dados, como se portou, os limites e como se usa.",
        PackageFileKind.MARKDOWN,
    ),
    files.MANIFEST: _PartInfo(
        "O manifesto",
        "A identidade do pacote, as versões das bibliotecas e a impressão digital de cada "
        "ficheiro.",
        PackageFileKind.JSON,
    ),
    files.SIGNATURE: _PartInfo(
        "A assinatura",
        "Diz se o pacote foi feito neste ML Lab e se não mudou desde então.",
        PackageFileKind.TEXT,
    ),
    files.INPUT: _PartInfo(
        "A entrada",
        "As colunas que o modelo espera, como chegam, com o que se viu no treino.",
        PackageFileKind.JSON,
    ),
    files.PREPARATION: _PartInfo(
        "A preparação",
        "A receita da preparação, o que ela aprendeu no treino e a linhagem.",
        PackageFileKind.JSON,
    ),
    files.RUN: _PartInfo(
        "O run",
        "A configuração, as métricas em cada parte, o que o modelo aprendeu e a curva da loss.",
        PackageFileKind.JSON,
    ),
    files.MODEL: _PartInfo(
        "O modelo",
        "O modelo treinado, para voltar a prever no ML Lab (joblib).",
        PackageFileKind.BINARY,
    ),
    files.ONNX: _PartInfo(
        "O modelo para outras apps",
        "A preparação e o modelo num só grafo ONNX, para o ONNX Runtime.",
        PackageFileKind.BINARY,
    ),
    files.ANIMATION: _PartInfo(
        "O treino passo a passo",
        "O que o modelo respondia em cada snapshot, para rever o treino.",
        PackageFileKind.JSON,
    ),
    files.NETWORK: _PartInfo(
        "As camadas da rede",
        "O que cada camada da rede fez em cada snapshot.",
        PackageFileKind.JSON,
    ),
    files.EXAMPLES: _PartInfo(
        "As linhas de exemplo",
        "Algumas linhas do teste, como chegaram, com o valor real e o previsto.",
        PackageFileKind.CSV,
    ),
}
"""O título, a descrição e o tipo de cada ficheiro, como o ecrã os mostra."""


@dataclass(frozen=True, slots=True)
class PackageCandidate:
    """Um run do histórico que se pode guardar num pacote (ou porque não se pode).

    Attributes:
        run_id: O run.
        name: O nome do run no histórico.
        model_display_name: O modelo.
        family: A família do modelo (a cor dele na interface).
        dataset_name: O dataset.
        task_type: A tarefa.
        created_at: Quando terminou.
        primary_label: O nome da métrica principal.
        primary_value: O valor dela na parte que resume o run.
        primary_as_percent: Se se mostra como percentagem.
        available: Se o modelo está à mão (desta sessão, ou guardado).
        reason: Porque não se pode guardar (``None`` se se pode).
    """

    run_id: str
    name: str
    model_display_name: str
    family: ModelFamily | None
    dataset_name: str
    task_type: TaskType
    created_at: datetime
    primary_label: str | None
    primary_value: float | None
    primary_as_percent: bool
    available: bool
    reason: str | None


type PackageField = Literal["include_model", "include_examples", "include_training"]
"""O campo do pedido (``PackageRequest``) que liga e desliga uma parte opcional."""


@dataclass(frozen=True, slots=True)
class PackageChoice:
    """Uma parte do pacote que se pode deixar de fora: o que leva e se este run a tem.

    Attributes:
        key: O campo do pedido (``PackageRequest``) que a liga e desliga.
        title: O nome a mostrar.
        description: O que leva, numa frase.
        included: Se vai, por omissão.
        available: Se este run a tem para dar.
        reason: Porque não se pode escolher (``None`` quando se pode).
    """

    key: PackageField
    title: str
    description: str
    included: bool
    available: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class PackageOptions:
    """O que se sugere para o pacote de um run, e o que se pode escolher.

    Attributes:
        run_id: O run.
        name: O nome sugerido.
        file_name: O nome do ficheiro que esse nome dá.
        choices: As partes que se podem deixar de fora, pela ordem em que se mostram (o
            modelo, as linhas de exemplo e o treino passo a passo).
        examples: Quantas linhas de exemplo, por omissão.
        max_examples: Quantas linhas de exemplo há, no máximo.
    """

    run_id: str
    name: str
    file_name: str
    choices: tuple[PackageChoice, ...]
    examples: int
    max_examples: int


@dataclass(frozen=True, slots=True)
class PackagePart:
    """Um ficheiro de um pacote, como o ecrã o mostra.

    Attributes:
        path: O nome no pacote.
        title: O nome a mostrar.
        description: O que tem, numa frase.
        kind: Como se pré-visualiza.
        size: O tamanho, em bytes, sem comprimir.
        content: O texto (os primeiros ``PREVIEW_CHARACTERS``); ``None`` num binário.
        truncated: Se o texto ficou cortado.
    """

    path: str
    title: str
    description: str
    kind: PackageFileKind
    size: int
    content: str | None
    truncated: bool


@dataclass(frozen=True, slots=True)
class PackagePreview:
    """Um pacote parte a parte: o que vai no ficheiro (ou o que um ficheiro tem).

    Attributes:
        name: O nome do pacote.
        file_name: O nome do ficheiro.
        format_version: A versão do formato.
        size: O tamanho de todas as partes, sem comprimir.
        parts: As partes, pela ordem do ZIP.
        has_model: Se leva o modelo (e por isso se pode prever com ele).
        has_onnx: Se leva o ``model.onnx``, para outras apps.
        notices: O que convém saber antes de o guardar ou de o abrir.
    """

    name: str
    file_name: str
    format_version: int
    size: int
    parts: tuple[PackagePart, ...]
    has_model: bool
    has_onnx: bool
    notices: tuple[Notice, ...]


@dataclass(frozen=True, slots=True)
class SavedPackage:
    """Um pacote da biblioteca.

    Attributes:
        package_id: O identificador na biblioteca.
        name: O nome do pacote.
        file_name: O nome do ficheiro.
        size: O tamanho do ficheiro (comprimido), em bytes.
        sha256: A impressão digital do ficheiro.
        created_at: Quando foi feito (do manifesto).
        added_at: Quando entrou na biblioteca.
        origin: Feito aqui, ou carregado.
        made_here: Se a assinatura é desta instalação.
        intact: Se os ficheiros conferem com o manifesto.
        trusted: Se o modelo se pode abrir (feito aqui, sem modelo, ou depois de confirmar).
        run_id: O run que o pacote pôs no histórico, se já pôs.
        model_display_name: O modelo.
        family: A família do modelo.
        dataset_name: O dataset.
        task_type: A tarefa.
        primary_label: O nome da métrica principal.
        primary_value: O valor dela.
        primary_as_percent: Se se mostra como percentagem.
        has_model: Se o pacote leva o modelo.
        has_onnx: Se leva o ``model.onnx``.
        notices: O que convém saber (de outra instalação, estragado, outras versões).
    """

    package_id: str
    name: str
    file_name: str
    size: int
    sha256: str
    created_at: datetime | None
    added_at: datetime
    origin: PackageOrigin
    made_here: bool
    intact: bool
    trusted: bool
    run_id: str | None
    model_display_name: str
    family: ModelFamily | None
    dataset_name: str
    task_type: TaskType | None
    primary_label: str | None
    primary_value: float | None
    primary_as_percent: bool
    has_model: bool
    has_onnx: bool
    notices: tuple[Notice, ...]


@dataclass(frozen=True, slots=True)
class PackageView:
    """Um pacote da biblioteca, aberto: o que se sabe dele e as suas partes.

    Attributes:
        package: O pacote.
        preview: As partes, no mesmo formato da pré-visualização.
    """

    package: SavedPackage
    preview: PackagePreview


@dataclass(frozen=True, slots=True)
class _Assembled:
    """Um pacote montado: os ficheiros, o nome e os avisos."""

    contents: PackageContents
    name: str
    file_name: str
    notices: tuple[Notice, ...]


class PackageService:
    """Os pacotes dos modelos: guardar, a biblioteca e carregar.

    Args:
        experiments: Onde estão o modelo e os dados dos últimos runs desta sessão.
        history: O histórico (reabre um run guardado e recebe os runs carregados).
        settings: As definições (a pasta dos artefactos, onde ficam a chave e a biblioteca).
        clock: A hora (para os testes poderem fixá-la).
    """

    def __init__(
        self,
        experiments: ExperimentService,
        history: HistoryService,
        settings: Settings | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._experiments = experiments
        self._history = history
        self._key = PackageKey(settings.artifacts_dir / "package.key")
        self._library = PackageLibrary(settings.artifacts_dir / "packages")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._onnx_cache: dict[str, tuple[bytes | None, str]] = {}
        self._onnx_lock = threading.Lock()

    def candidates(self) -> tuple[PackageCandidate, ...]:
        """Os runs do histórico que se podem guardar num pacote, do mais recente ao mais antigo.

        Os de reinforcement learning ficam de fora: não têm preparação.
        """
        found = []
        for entry in self._history.runs():
            if entry.task_type is TaskType.CONTROL:
                continue
            available = entry.saved or self._in_memory(entry.run_id)
            found.append(
                PackageCandidate(
                    run_id=entry.run_id,
                    name=entry.name,
                    model_display_name=entry.model_display_name,
                    family=_family(entry.model_name),
                    dataset_name=entry.dataset_name,
                    task_type=entry.task_type,
                    created_at=entry.created_at,
                    primary_label=entry.primary_label,
                    primary_value=entry.primary_value,
                    primary_as_percent=entry.primary_as_percent,
                    available=available,
                    reason=None
                    if available
                    else (
                        "Este run só ficou registado, sem o modelo: reproduz-o no Histórico "
                        "para o guardar num pacote."
                    ),
                )
            )
        return tuple(found)

    def options(self, run_id: str) -> PackageOptions:
        """O nome e as partes sugeridas para o pacote de um run.

        Raises:
            TrainingInProgressError: Se o run ainda estiver a treinar.
            RunNotFoundError: Se o run não estiver no histórico.
            RunNotInMemoryError: Se o run não tiver o modelo à mão (só registado).
        """
        artifacts = self._artifacts(run_id)
        preparation = _preparation(artifacts)
        name = _default_name(artifacts.result)
        return PackageOptions(
            run_id=run_id,
            name=name,
            file_name=_file_name(name),
            choices=_choices(artifacts, preparation),
            examples=min(5, len(preparation.examples)) or 1,
            max_examples=max(1, len(preparation.examples)),
        )

    def preview(self, run_id: str, request: PackageRequest) -> PackagePreview:
        """O pacote de um run, parte a parte, antes de existir.

        É exatamente o que ``save`` grava, no mesmo instante: o mesmo código monta os dois.

        Raises:
            TrainingInProgressError: Se o run ainda estiver a treinar.
            RunNotFoundError: Se o run não estiver no histórico.
            RunNotInMemoryError: Se o run não tiver o modelo à mão (só registado).
        """
        assembled = self._assemble(self._artifacts(run_id), request, self._clock())
        return _preview(assembled)

    def save(self, run_id: str, request: PackageRequest) -> SavedPackage:
        """Grava o pacote de um run na biblioteca e guarda o run no histórico (com o modelo).

        Raises:
            TrainingInProgressError: Se o run ainda estiver a treinar.
            RunNotFoundError: Se o run não estiver no histórico.
            RunNotInMemoryError: Se o run não tiver o modelo à mão (só registado).
        """
        artifacts = self._artifacts(run_id)
        assembled = self._assemble(artifacts, request, self._clock())
        # O run também fica guardado no histórico, como fazia o botão do treino (ADR 0039).
        self._history.save(run_id)
        data = assembled.contents.to_zip()
        opened = files.read_package(data, self._key)
        entry = self._library.add(
            data,
            file_name=assembled.file_name,
            origin="made_here",
            trusted=True,
            opened=opened,
            run_id=run_id,
        )
        return self._saved(entry)

    def saved(self) -> tuple[SavedPackage, ...]:
        """Os pacotes da biblioteca, do mais recente para o mais antigo."""
        return tuple(self._saved(entry) for entry in self._library.entries())

    def contents(self, package_id: str) -> PackageView:
        """Um pacote da biblioteca, aberto parte a parte.

        Raises:
            PackageNotFoundError: Se não houver nenhum com este identificador.
        """
        entry = self._library.entry(package_id)
        opened = files.read_package(self._library.data(package_id), self._key)
        package = self._saved(entry)
        assembled = _Assembled(opened.contents, package.name, entry.file_name, package.notices)
        return PackageView(package=package, preview=_preview(assembled))

    def load(self, filename: str, content: bytes) -> SavedPackage:
        """Carrega um ``.mllab``: entra na biblioteca e, se não houver risco, no histórico.

        Um pacote feito aqui, ou sem o modelo, entra logo no histórico. Um de outra
        instalação fica na biblioteca só para ler, até a pessoa confirmar que confia nele
        (``trust``): abrir o modelo corre o código de quem o fez.

        Raises:
            PackageError: Se o ficheiro não for um pacote, ou for de um formato mais novo.
        """
        opened = files.read_package(content, self._key)
        known = self._library.find(_sha256(content))
        if known is not None:
            return self._saved(known)
        has_model = opened.contents.get(files.MODEL) is not None
        safe = opened.intact and (opened.made_here or not has_model)
        entry = self._library.add(
            content, file_name=filename, origin="loaded", trusted=safe, opened=opened
        )
        if safe:
            model = files.load_model(opened) if has_model else None
            entry = self._library.update(entry.package_id, run_id=self._adopt(entry, opened, model))
        return self._saved(entry)

    def trust(self, package_id: str) -> SavedPackage:
        """Abre o modelo de um pacote de outra instalação e põe o run no histórico.

        É o passo depois do aviso: a pessoa confirmou que confia em quem lhe deu o ficheiro.

        Raises:
            PackageNotFoundError: Se não houver nenhum com este identificador.
            PackageError: Se o pacote estiver estragado ou alterado (nunca abre).
            TrackingError: Se o modelo não abrir (por exemplo, de outra versão das bibliotecas).
        """
        entry = self._library.entry(package_id)
        if entry.trusted and entry.run_id is not None:
            return self._saved(entry)
        opened = files.read_package(self._library.data(package_id), self._key)
        has_model = opened.contents.get(files.MODEL) is not None
        model = files.load_model(opened) if has_model else None
        run_id = self._adopt(entry, opened, model)
        return self._saved(self._library.update(package_id, trusted=True, run_id=run_id))

    def delete(self, package_id: str) -> None:
        """Apaga um pacote da biblioteca (o run que pôs no histórico fica lá)."""
        self._library.delete(package_id)

    @internal
    def file(self, package_id: str) -> tuple[str, bytes]:
        """O nome e os bytes de um pacote, para o descarregar (só a API o usa).

        Raises:
            PackageNotFoundError: Se não houver nenhum com este identificador.
        """
        entry = self._library.entry(package_id)
        return entry.file_name, self._library.data(package_id)

    def _artifacts(self, run_id: str) -> RunArtifacts:
        """O modelo, os dados e o resultado de um run; um run guardado volta à memória.

        Raises:
            TrainingInProgressError: Se o run ainda estiver a treinar.
            TrainingFailedError: Se o treino do run falhou.
            RunNotFoundError: Se o run não estiver no histórico.
            RunNotInMemoryError: Se o run só tiver ficado registado, sem o modelo.
        """
        try:
            return self._experiments.artifacts(run_id)
        except RunNotInMemoryError:
            pass
        try:
            # Um treino que ainda decorre (ou que falhou) diz porquê, em vez de "não está no
            # histórico"; um que acabou entretanto já está à mão.
            self._experiments.result(run_id)
            return self._experiments.artifacts(run_id)
        except RunNotInMemoryError:
            entry = self._history.entry(run_id)
            if not entry.saved:
                raise
            self._history.open(run_id)  # reabre-o com o modelo e os dados
            return self._experiments.artifacts(run_id)

    def _in_memory(self, run_id: str) -> bool:
        try:
            self._experiments.artifacts(run_id)
        except RunNotInMemoryError:
            return False
        return True

    def _assemble(
        self, artifacts: RunArtifacts, request: PackageRequest, created_at: datetime
    ) -> _Assembled:
        """Os ficheiros de um pacote: o que a pré-visualização mostra e o que o ZIP leva."""
        result = artifacts.result
        preparation = _preparation(artifacts)
        info = MODELS.get(result.model_name).info
        name = (request.name or "").strip() or _default_name(result)
        examples = request.include_examples
        if examples is None:
            examples = artifacts.config.dataset.kind in _EXAMPLE_SOURCES
        examples = examples and _examples_available(artifacts, preparation)
        # O ONNX só vai com o modelo; sem ele, nem se converte.
        onnx, onnx_reason = self._onnx(artifacts) if request.include_model else (None, "")

        parts = [
            PackageFile(files.INPUT, _json(preparation.input_document())),
            PackageFile(
                files.PREPARATION,
                _json(
                    {
                        **preparation.preparation_document(),
                        "lineage": lineage_document(artifacts.dataset.lineage),
                    }
                ),
            ),
            PackageFile(files.RUN, _json(to_json(replace(result, animation=None, network=None)))),
        ]
        if request.include_model:
            parts.append(PackageFile(files.MODEL, files.model_bytes(artifacts.model)))
            if onnx is not None:
                parts.append(PackageFile(files.ONNX, onnx))
        if request.include_training:
            if result.animation is not None:
                parts.append(PackageFile(files.ANIMATION, _json(to_json(result.animation))))
            if result.network is not None:
                parts.append(PackageFile(files.NETWORK, _json(to_json(result.network))))
        if examples:
            predicted = _predictions(artifacts, request.examples)
            text = preparation.examples_csv(predicted, request.examples)
            parts.append(PackageFile(files.EXAMPLES, text.encode("utf-8")))

        present = [*(part.path for part in parts), files.README, files.MANIFEST]
        card_files = [
            CardFile(path, _PARTS[path].description)
            for path in files.FILE_ORDER
            if path in present or path == files.SIGNATURE
        ]
        readme = model_card(
            name=name,
            result=result,
            dataset=artifacts.dataset,
            preparation=preparation,
            info=info,
            files=card_files,
            onnx=None if onnx is None else onnx_signature(onnx, preparation),
            onnx_reason=onnx_reason if request.include_model and onnx is None else None,
        )
        parts.append(PackageFile(files.README, readme.encode("utf-8")))
        manifest = _manifest(
            name, result, artifacts, info, created_at, request, examples, onnx is not None
        )
        contents = files.assemble(parts, manifest, self._key)
        notices = _notices(result, info, contents, request, examples, onnx_reason, name)
        return _Assembled(contents, name, _file_name(name), notices)

    def _onnx(self, artifacts: RunArtifacts) -> tuple[bytes | None, str]:
        """O ``model.onnx`` de um run (já confirmado), ou a razão por que não o há.

        Converter e confirmar demora (o skl2onnx, o ONNX Runtime), e a pré-visualização pede-se
        a cada opção que muda: o resultado fica guardado, para os últimos runs.
        """
        run_id = artifacts.result.run_id
        with self._onnx_lock:
            cached = self._onnx_cache.get(run_id)
        if cached is not None:
            return cached
        exported: tuple[bytes | None, str]
        try:
            exported = export_onnx(artifacts.model, _preparation(artifacts)), ""
        except CapabilityError as error:
            exported = None, str(error)
        # O ONNX é um extra: uma conversão que rebenta de outra maneira não impede o pacote.
        except Exception:
            logger.exception("a conversão para ONNX falhou", extra={"run_id": run_id})
            exported = None, "A conversão para ONNX falhou com um erro inesperado."
        with self._onnx_lock:
            self._onnx_cache[run_id] = exported
            while len(self._onnx_cache) > ONNX_CACHE:
                self._onnx_cache.pop(next(iter(self._onnx_cache)))
        return exported

    def _adopt(self, entry: LibraryEntry, opened: OpenedPackage, model: Any | None) -> str:
        """Põe no histórico o run de um pacote carregado; devolve o identificador dele."""
        result = from_json(ExperimentResult, opened.contents.document(files.RUN))
        try:
            self._history.entry(result.run_id)
            exists = True
        except RunNotFoundError:
            exists = False
        if exists and opened.made_here:
            return result.run_id  # é um run deste histórico, que voltou num pacote
        if exists:
            result = replace(result, run_id=f"run-{secrets.token_hex(4)}")
        manifest = opened.manifest
        name = str(manifest.get("name") or result.model_display_name)
        where = "neste ML Lab" if opened.made_here else "noutro ML Lab"
        notice = Notice.info(
            "loaded_from_package",
            f"Este run veio do pacote «{entry.file_name}», feito {where} e carregado a "
            f"{entry.added_at:%d/%m/%Y}.",
        )
        result = replace(result, notices=(notice, *result.notices))
        documents = {
            name_: opened.contents.document(path)
            for name_, path in (("animation", files.ANIMATION), ("network", files.NETWORK))
            if opened.contents.get(path) is not None
        }
        _check_documents(documents)
        self._history.adopt(result, model, documents, name=f"{name} (carregado)")
        return result.run_id

    def _saved(self, entry: LibraryEntry) -> SavedPackage:
        manifest = entry.manifest
        model: Mapping[str, Any] = _mapping(manifest.get("model"))
        metric: Mapping[str, Any] = _mapping(manifest.get("primary_metric"))
        listed = {
            str(item.get("path")) for item in manifest.get("files", []) if isinstance(item, Mapping)
        }
        notices: list[Notice] = []
        if not entry.intact:
            notices.append(
                Notice.error(
                    "package_damaged",
                    "O pacote está estragado ou foi alterado depois de feito: "
                    f"{' '.join(entry.problems)} O modelo não abre.",
                )
            )
        elif not entry.trusted:
            notices.append(
                Notice.warning(
                    "package_from_elsewhere",
                    "Este modelo foi feito noutro ML Lab. Vem num ficheiro joblib, e abri-lo "
                    "corre o código de quem o fez: só o abras se confias em quem to deu. A "
                    "ficha, a avaliação e a preparação já estão à vista, e lê-las não tem risco.",
                )
            )
        notices.extend(_version_notices(manifest))
        return SavedPackage(
            package_id=entry.package_id,
            name=str(manifest.get("name") or entry.file_name),
            file_name=entry.file_name,
            size=entry.size,
            sha256=entry.sha256,
            created_at=_moment(manifest.get("created_at")),
            added_at=entry.added_at,
            origin=PackageOrigin(entry.origin),
            made_here=entry.made_here,
            intact=entry.intact,
            trusted=entry.trusted,
            run_id=entry.run_id,
            model_display_name=str(model.get("display_name", "")),
            family=_family_value(model.get("family")),
            dataset_name=str(_get(manifest, "dataset", "name") or ""),
            task_type=_task(manifest.get("task_type")),
            primary_label=_text(metric.get("label")),
            primary_value=_number(metric.get("value")),
            primary_as_percent=bool(metric.get("as_percent", False)),
            has_model=files.MODEL in listed,
            has_onnx=files.ONNX in listed,
            notices=tuple(notices),
        )


def _preview(assembled: _Assembled) -> PackagePreview:
    parts = []
    for file in assembled.contents.files:
        info = _PARTS[file.path]
        content: str | None = None
        truncated = False
        if info.kind is not PackageFileKind.BINARY:
            text = file.content.decode("utf-8", errors="replace")
            truncated = len(text) > PREVIEW_CHARACTERS
            content = text[:PREVIEW_CHARACTERS]
        parts.append(
            PackagePart(
                path=file.path,
                title=info.title,
                description=info.description,
                kind=info.kind,
                size=len(file.content),
                content=content,
                truncated=truncated,
            )
        )
    paths = {file.path for file in assembled.contents.files}
    return PackagePreview(
        name=assembled.name,
        file_name=assembled.file_name,
        format_version=files.FORMAT_VERSION,
        size=assembled.contents.size,
        parts=tuple(parts),
        has_model=files.MODEL in paths,
        has_onnx=files.ONNX in paths,
        notices=assembled.notices,
    )


def _manifest(
    name: str,
    result: ExperimentResult,
    artifacts: RunArtifacts,
    info: ModelInfo,
    created_at: datetime,
    request: PackageRequest,
    examples: bool,
    has_onnx: bool,
) -> dict[str, Any]:
    primary = result.primary
    dataset = artifacts.dataset
    return {
        "format": files.FORMAT,
        "format_version": files.FORMAT_VERSION,
        "name": name,
        "created_at": created_at.replace(microsecond=0).isoformat(),
        "run_id": result.run_id,
        "seed": result.seed,
        "task_type": result.task_type.value,
        "target": dataset.target_name,
        "dataset": {
            "name": result.dataset_name,
            "source": _source(artifacts.config.dataset),
            "rows": {part.value: len(split) for part, split in dataset.splits.items()},
        },
        "model": {
            "name": info.name,
            "display_name": info.display_name,
            "family": info.family.value,
            "implementation": info.implementation.value,
            "hyperparameters": to_json(result.hyperparameters),
        },
        "primary_metric": None
        if primary is None
        else {
            "name": primary.name,
            "label": primary.display_name,
            "partition": result.main_partition.value,
            "value": primary.value,
            "as_percent": primary.as_percent,
        },
        "contents": {
            "model": request.include_model,
            "onnx": has_onnx and request.include_model,
            "examples": request.examples if examples else 0,
            "training": request.include_training,
        },
        "versions": _versions(),
    }


def _notices(
    result: ExperimentResult,
    info: ModelInfo,
    contents: PackageContents,
    request: PackageRequest,
    examples: bool,
    onnx_reason: str,
    name: str,
) -> tuple[Notice, ...]:
    """O que convém saber antes de guardar o pacote."""
    notices: list[Notice] = []
    if examples:
        notices.append(
            Notice.warning(
                "package_real_rows",
                f"As linhas de exemplo são dados reais de {result.dataset_name}: quem tiver o "
                "ficheiro vê-as.",
            )
        )
    if request.include_model and info.keeps_training_rows:
        notices.append(
            Notice.warning(
                "package_memorized_rows",
                f"{info.display_name} guarda linhas do treino dentro dele (já transformadas): "
                "quem tiver o modelo consegue vê-las.",
            )
        )
    if not info.supports_new_data:
        notices.append(
            Notice.info(
                "package_cannot_predict",
                f"{info.display_name} não prevê linhas novas: o pacote serve para ler o que o "
                "modelo fez, não para prever.",
            )
        )
    if not request.include_model:
        notices.append(
            Notice.info(
                "package_without_model",
                "Sem o modelo, o pacote lê-se todo (a ficha, a avaliação, a preparação), mas não "
                "prevê. Em troca, não tem pickle nenhum e pode partilhar-se à vontade.",
            )
        )
    elif contents.get(files.ONNX) is None:
        notices.append(Notice.info("package_without_onnx", onnx_reason))
    if contents.size > LARGE_PACKAGE:
        notices.append(
            Notice.warning(
                "package_large",
                f"O pacote «{name}» tem {contents.size // (1024 * 1024)} MB: guardar e "
                "descarregar demora, e partilhá-lo também.",
            )
        )
    return tuple(notices)


def _version_notices(manifest: Mapping[str, Any]) -> list[Notice]:
    """Um aviso por biblioteca de que o pickle depende que tenha outra versão aqui."""
    recorded = manifest.get("versions")
    if not isinstance(recorded, Mapping) or not manifest.get("contents", {}).get("model", True):
        return []
    current = _versions()
    different = [
        f"{name} {recorded[name]} (aqui, {current[name]})"
        for name in _PICKLED
        if recorded.get(name)
        and current.get(name)
        and _minor(recorded[name]) != _minor(current[name])
    ]
    if not different:
        return []
    return [
        Notice.warning(
            "package_other_versions",
            f"O pacote foi feito com outras versões: {', '.join(different)}. O modelo pode não "
            "abrir, ou prever de outra maneira.",
        )
    ]


def _check_documents(documents: Mapping[str, Any]) -> None:
    """Confirma que o treino passo a passo de um pacote se lê, antes de o guardar."""
    kinds = {"animation": TrainingAnimation, "network": NetworkTrace}
    for name, document in documents.items():
        from_json(kinds[name], document)


def _preparation(artifacts: RunArtifacts) -> FittedPreparation:
    preparation = artifacts.dataset.preparation
    if preparation is None:
        raise PackageError(
            "Os dados deste run não guardaram a preparação aprendida: reproduz o run para o "
            "guardar num pacote."
        )
    return preparation


def _choices(artifacts: RunArtifacts, preparation: FittedPreparation) -> tuple[PackageChoice, ...]:
    """As partes que se podem deixar de fora do pacote de um run, e se ele as tem."""
    result = artifacts.result
    examples = _examples_reason(artifacts, preparation)
    training = result.animation is not None or result.network is not None
    return (
        PackageChoice(
            key="include_model",
            title="O modelo, para prever",
            description=(
                "O modelo treinado, para voltar a prever no ML Lab (e, quando há conversão, "
                "noutras apps, em ONNX). Sem ele, o pacote só se lê."
            ),
            included=True,
            available=True,
            reason=None,
        ),
        PackageChoice(
            key="include_examples",
            title="Linhas de exemplo",
            description=(
                f"{_PARTS[files.EXAMPLES].description} Servem para confirmar que o modelo "
                "prevê o mesmo noutro sítio."
            ),
            # Por omissão, só as dos datasets de exemplo e dos sintéticos: as de um ficheiro
            # carregado podem ser de alguém.
            included=examples is None and artifacts.config.dataset.kind in _EXAMPLE_SOURCES,
            available=examples is None,
            reason=examples,
        ),
        PackageChoice(
            key="include_training",
            title="O treino passo a passo",
            description=(
                "O que o modelo respondia em cada snapshot (e o que cada camada da rede fez), "
                "para rever o treino. Pode ocupar bastante."
            ),
            included=False,
            available=training,
            reason=None
            if training
            else (
                "Este run não tem o treino passo a passo: os dados não têm uma projeção 2D, ou "
                "o run voltou do histórico sem ele."
            ),
        ),
    )


def _examples_reason(artifacts: RunArtifacts, preparation: FittedPreparation) -> str | None:
    """Porque é que um run não tem linhas de exemplo para dar (``None`` se as tem)."""
    info = MODELS.get(artifacts.result.model_name).info
    if (
        not info.supports_new_data
        or artifacts.result.task_type is TaskType.DIMENSIONALITY_REDUCTION
    ):
        return "Este modelo não prevê linhas novas: não há exemplos com a previsão."
    if len(preparation.examples) == 0:
        return "O teste deste run não ficou com linhas para dar de exemplo."
    return None


def _examples_available(artifacts: RunArtifacts, preparation: FittedPreparation) -> bool:
    return _examples_reason(artifacts, preparation) is None


def _predictions(artifacts: RunArtifacts, n_rows: int) -> list[object]:
    """O que o modelo prevê para as primeiras linhas do teste (as classes pelo nome)."""
    rows = artifacts.dataset.test.X[:n_rows]
    predicted = np.asarray(artifacts.model.predict(rows))
    labels = artifacts.dataset.class_labels
    if artifacts.result.task_type.is_classification and labels:
        return [labels[int(value)] for value in predicted]
    return [value.item() if isinstance(value, np.generic) else value for value in predicted]


def _versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in _VERSIONED:
        try:
            found[name] = version(name)
        except DistributionNotFound:
            found[name] = None
    return found


def _minor(text: object) -> str:
    """A versão até ao segundo número (``1.9.1`` dá ``1.9``)."""
    return ".".join(str(text).split(".")[:2])


def _source(spec: SourceSpec) -> dict[str, Any]:
    return dict(to_json(spec.to_dict()))


def _json(document: Any) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _default_name(result: ExperimentResult) -> str:
    return f"{result.dataset_name} — {result.model_display_name}"


def _file_name(name: str) -> str:
    """O nome do ficheiro de um pacote: o nome sem acentos, em minúsculas e com hífenes."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
    return f"{slug or 'modelo'}{files.EXTENSION}"


def _family(model_name: str) -> ModelFamily | None:
    try:
        return MODELS.get(model_name).info.family
    except LookupError:
        return None


def _family_value(value: object) -> ModelFamily | None:
    try:
        return ModelFamily(str(value))
    except ValueError:
        return None


def _task(value: object) -> TaskType | None:
    try:
        return TaskType(str(value))
    except ValueError:
        return None


def _moment(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _get(document: Mapping[str, Any], *keys: str) -> Any:
    current: Any = document
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current
