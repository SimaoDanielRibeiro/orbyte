"""O pacote de um modelo: um ficheiro ``.mllab`` para o levar e voltar a abrir (ADR 0039).

Um ``.mllab`` é um ZIP com nomes fixos. Abre-se com qualquer programa de ZIP, e o formato
está descrito em ``docs/formato-mllab.md``, para outras apps o saberem ler. Este módulo sabe:

- **montar** os ficheiros de um pacote: junta ao manifesto o tamanho e o SHA-256 de cada
  ficheiro e assina-o (HMAC-SHA256, com a chave desta instalação). Os ficheiros montados são
  os que a pré-visualização mostra e os que vão para o ZIP: a pré-visualização é o ficheiro;
- **ler** um pacote: só os nomes conhecidos, com os tamanhos e as impressões digitais
  confirmados, sem extrair nada para o disco. A assinatura diz se foi feito aqui;
- **abrir o modelo** de um pacote. O ``model.joblib`` é um pickle: abri-lo corre o código de
  quem o fez. Quem chama decide se pode (o serviço só o faz num pacote feito aqui, ou depois
  de a pessoa confirmar que confia nele);
- **guardar** pacotes numa biblioteca, uma pasta por pacote.

O joblib só se usa em ``core/tracking/`` (há uma regra de arquitetura que o confirma).
"""

import hashlib
import hmac
import io
import json
import re
import secrets
import shutil
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import joblib

from ml_lab.core.foundation.errors import PackageError, PackageNotFoundError, TrackingError

FORMAT: Final = "ml-lab-package"
"""O nome do formato, no manifesto."""

FORMAT_VERSION: Final = 1
"""A versão do formato que este laboratório escreve (e a mais nova que sabe ler)."""

EXTENSION: Final = ".mllab"

README: Final = "README.md"
MANIFEST: Final = "manifest.json"
SIGNATURE: Final = "manifest.sig"
INPUT: Final = "input.json"
PREPARATION: Final = "preparation.json"
RUN: Final = "run.json"
MODEL: Final = "model.joblib"
ONNX: Final = "model.onnx"
ANIMATION: Final = "animation.json"
NETWORK: Final = "network.json"
EXAMPLES: Final = "examples.csv"

FILE_ORDER: Final = (
    README,
    MANIFEST,
    SIGNATURE,
    INPUT,
    PREPARATION,
    RUN,
    MODEL,
    ONNX,
    ANIMATION,
    NETWORK,
    EXAMPLES,
)
"""Os ficheiros que um pacote pode ter, pela ordem em que vão no ZIP (e se mostram)."""

REQUIRED: Final = (README, MANIFEST, INPUT, PREPARATION, RUN)
"""Os ficheiros sem os quais um pacote não se lê."""

MAX_PACKAGE_BYTES: Final = 512 * 1024 * 1024
"""O maior pacote que se lê (um ZIP maior é recusado antes de abrir)."""

_SAFE_ID: Final = re.compile(r"pkg-[0-9a-f]{8}")
_LIBRARY_FILE: Final = "library.json"
_PACKAGE_FILE: Final = "package.mllab"

type Origin = Literal["made_here", "loaded"]
"""De onde veio um pacote da biblioteca: feito aqui, ou carregado de um ficheiro."""


@dataclass(frozen=True, slots=True)
class PackageFile:
    """Um ficheiro de um pacote.

    Attributes:
        path: O nome dentro do ZIP (um dos de ``FILE_ORDER``).
        content: Os bytes.
    """

    path: str
    content: bytes

    @property
    def sha256(self) -> str:
        """A impressão digital do conteúdo."""
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class PackageContents:
    """Os ficheiros de um pacote, pela ordem do ZIP, com o manifesto e a assinatura.

    Attributes:
        files: Os ficheiros.
    """

    files: tuple[PackageFile, ...]

    def get(self, path: str) -> bytes | None:
        """O conteúdo de um ficheiro, ou ``None`` se o pacote não o tiver."""
        return next((file.content for file in self.files if file.path == path), None)

    def text(self, path: str) -> str | None:
        """O conteúdo de um ficheiro de texto, ou ``None`` se o pacote não o tiver."""
        content = self.get(path)
        return None if content is None else content.decode("utf-8")

    def document(self, path: str) -> Any:
        """Um ficheiro JSON, já lido (``None`` se o pacote não o tiver).

        Raises:
            PackageError: Se o ficheiro não for JSON.
        """
        text = self.text(path)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise PackageError(f"The package file {path} isn't JSON: {error}.") from error

    @property
    def manifest(self) -> dict[str, Any]:
        """O manifesto."""
        manifest = self.document(MANIFEST)
        if not isinstance(manifest, dict):
            raise PackageError("The package has no valid manifest (manifest.json).")
        return manifest

    @property
    def size(self) -> int:
        """O tamanho de todos os ficheiros, sem comprimir (o ZIP comprime alguns)."""
        return sum(len(file.content) for file in self.files)

    def to_zip(self) -> bytes:
        """O ficheiro ``.mllab``: um ZIP com os ficheiros, comprimidos."""
        created = _zip_time(self.manifest.get("created_at"))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in self.files:
                info = zipfile.ZipInfo(file.path, date_time=created)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, file.content)
        return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class OpenedPackage:
    """Um pacote lido de um ficheiro.

    Attributes:
        contents: Os ficheiros.
        made_here: Se a assinatura é desta instalação (foi feito aqui e não mudou).
        intact: Se todos os ficheiros conferem com o manifesto.
        problems: O que não confere, em frases (vazio num pacote intacto).
    """

    contents: PackageContents
    made_here: bool
    intact: bool
    problems: tuple[str, ...] = ()

    @property
    def manifest(self) -> dict[str, Any]:
        """O manifesto."""
        return self.contents.manifest


class PackageKey:
    """A chave desta instalação, com que se assinam os pacotes feitos aqui.

    São 32 bytes ao acaso, criados no primeiro uso. Quem tem a chave consegue fazer pacotes
    que esta instalação abre sem perguntar; por isso fica num ficheiro só legível pelo dono.

    Args:
        path: O ficheiro da chave (normalmente ``artifacts/package.key``).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def sign(self, data: bytes) -> str:
        """A assinatura de ``data`` (HMAC-SHA256, em hexadecimal); cria a chave se faltar."""
        key = self._existing()
        if key is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            key = secrets.token_bytes(32)
            self.path.write_bytes(key)
            self.path.chmod(0o600)
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        """Se ``signature`` é a assinatura desta instalação para ``data``."""
        key = self._existing()
        if key is None:
            return False
        expected = hmac.new(key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature.strip())

    def _existing(self) -> bytes | None:
        return self.path.read_bytes() if self.path.is_file() else None


def model_bytes(model: Any) -> bytes:
    """O modelo treinado em joblib (comprimido), como fica no ``model.joblib`` de um pacote."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.getvalue()


def assemble(
    files: Sequence[PackageFile], manifest: Mapping[str, Any], key: PackageKey
) -> PackageContents:
    """Monta um pacote: o manifesto com o tamanho e o SHA-256 de cada ficheiro, e a assinatura.

    Args:
        files: Os ficheiros do pacote, sem o manifesto nem a assinatura.
        manifest: O manifesto, sem a lista dos ficheiros (que se junta aqui).
        key: A chave desta instalação.

    Raises:
        PackageError: Se um ficheiro tiver um nome que um pacote não pode ter, ou se faltar
            um dos obrigatórios.
    """
    names = [file.path for file in files]
    unknown = [name for name in names if name not in FILE_ORDER or name in (MANIFEST, SIGNATURE)]
    missing = [name for name in REQUIRED if name not in (*names, MANIFEST)]
    if unknown or missing:
        raise PackageError(
            f"A package can't have {unknown} and must have {missing}: those are the format's names."
        )
    listed = [
        {"path": file.path, "bytes": len(file.content), "sha256": file.sha256}
        for file in _ordered(files)
    ]
    text = json.dumps({**manifest, "files": listed}, ensure_ascii=False, indent=2)
    manifest_file = PackageFile(MANIFEST, (text + "\n").encode("utf-8"))
    signature = PackageFile(SIGNATURE, (key.sign(manifest_file.content) + "\n").encode("ascii"))
    return PackageContents(_ordered([*files, manifest_file, signature]))


def read_package(data: bytes, key: PackageKey) -> OpenedPackage:
    """Lê um pacote, confirmando cada ficheiro contra o manifesto.

    Um pacote de outra instalação lê-se como um feito aqui (``made_here`` diz a diferença).
    Um ficheiro estragado também se lê, se der, e ``problems`` diz o que não confere.

    Raises:
        PackageError: Se não for um ZIP, for grande demais, não tiver um manifesto do formato
            ou for de uma versão do formato mais nova do que esta.
    """
    if len(data) > MAX_PACKAGE_BYTES:
        raise PackageError(
            f"The file is {len(data) // (1024 * 1024)} MB: the largest package that opens is "
            f"{MAX_PACKAGE_BYTES // (1024 * 1024)} MB."
        )
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise PackageError(
            "This file isn't an Orbyte package: a .mllab is a ZIP, and this one doesn't open as "
            "one."
        ) from error
    problems: list[str] = []
    files: list[PackageFile] = []
    total = 0
    with archive:
        for info in archive.infolist():
            if info.filename not in FILE_ORDER:
                problems.append(f"The package has a file the format doesn't know: {info.filename}.")
                continue
            total += info.file_size
            if total > MAX_PACKAGE_BYTES:
                raise PackageError("The package, uncompressed, is too large to open.")
            try:
                content = archive.read(info)
            except (zipfile.BadZipFile, zlib.error, EOFError):
                # Os bytes do ZIP estragaram-se (a impressão do próprio ZIP não confere): o resto
                # lê-se na mesma, e o manifesto diz que falta este.
                problems.append(f"The file {info.filename} is damaged and can't be read.")
                continue
            files.append(PackageFile(info.filename, content))
    contents = PackageContents(_ordered(files))
    manifest = _manifest_of(contents)
    problems.extend(_check(contents, manifest))
    signature = contents.text(SIGNATURE)
    manifest_bytes = contents.get(MANIFEST) or b""
    made_here = signature is not None and key.verify(manifest_bytes, signature) and not problems
    return OpenedPackage(
        contents, made_here=made_here, intact=not problems, problems=tuple(problems)
    )


def load_model(opened: OpenedPackage) -> Any:
    """O modelo de um pacote (corre o pickle: quem chama tem de confiar no pacote).

    Raises:
        PackageError: Se o pacote não tiver o modelo, ou estiver estragado.
        TrackingError: Se o modelo não abrir (por exemplo, de outra versão das bibliotecas).
    """
    if not opened.intact:
        raise PackageError(
            "The package is damaged or was changed after it was made, so the model won't "
            f"open: {' '.join(opened.problems)}"
        )
    content = opened.contents.get(MODEL)
    if content is None:
        raise PackageError(
            "This package doesn't carry the model (it was saved with the card only)."
        )
    try:
        return joblib.load(io.BytesIO(content))
    except Exception as error:  # um pickle de outra versão do código falha de muitas formas
        raise TrackingError(f"The package's model didn't open: {error}") from error


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    """Um pacote da biblioteca.

    Attributes:
        package_id: O identificador na biblioteca.
        file_name: O nome do ficheiro (o que se descarrega).
        sha256: A impressão digital do ficheiro.
        size: O tamanho do ficheiro, em bytes.
        added_at: Quando entrou na biblioteca (UTC).
        origin: Feito aqui, ou carregado de um ficheiro.
        trusted: Se o modelo se pode abrir (feito aqui, ou a pessoa confirmou que confia).
        run_id: O run que o pacote pôs no histórico desta instalação, se já pôs.
        made_here: Se a assinatura é desta instalação (lido quando entrou na biblioteca).
        intact: Se todos os ficheiros conferiam com o manifesto.
        problems: O que não conferia.
        manifest: O manifesto, para mostrar o pacote sem voltar a abrir o ZIP.
    """

    package_id: str
    file_name: str
    sha256: str
    size: int
    added_at: datetime
    origin: Origin
    trusted: bool
    run_id: str | None = None
    made_here: bool = False
    intact: bool = True
    problems: tuple[str, ...] = ()
    manifest: Mapping[str, Any] = field(default_factory=dict)


class PackageLibrary:
    """A biblioteca dos pacotes: uma pasta por pacote, com o ficheiro e o que se sabe dele.

    Args:
        directory: A pasta (normalmente ``artifacts/packages``); é criada quando faz falta.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def add(
        self,
        data: bytes,
        *,
        file_name: str,
        origin: Origin,
        trusted: bool,
        opened: OpenedPackage | None = None,
        package_id: str | None = None,
        run_id: str | None = None,
    ) -> LibraryEntry:
        """Guarda um pacote na biblioteca.

        Args:
            data: O ficheiro ``.mllab``.
            file_name: O nome do ficheiro.
            origin: Feito aqui, ou carregado.
            trusted: Se o modelo se pode abrir.
            opened: O pacote já lido (o manifesto e se confere), para a biblioteca o mostrar
                sem voltar a abrir o ZIP.
            package_id: O identificador a usar; se faltar, ou já existir, cria-se outro.
            run_id: O run do histórico a que o pacote corresponde.
        """
        identifier = package_id
        if identifier is None or not _SAFE_ID.fullmatch(identifier) or self._exists(identifier):
            identifier = new_package_id()
            while self._exists(identifier):
                identifier = new_package_id()
        entry = LibraryEntry(
            package_id=identifier,
            file_name=_file_name(file_name),
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            added_at=datetime.now(UTC),
            origin=origin,
            trusted=trusted,
            run_id=run_id,
            made_here=opened is not None and opened.made_here,
            intact=opened is None or opened.intact,
            problems=() if opened is None else opened.problems,
            manifest={} if opened is None else opened.manifest,
        )
        folder = self._folder(identifier)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / _PACKAGE_FILE).write_bytes(data)
        self._write(entry)
        return entry

    def entries(self) -> tuple[LibraryEntry, ...]:
        """Todos os pacotes, do mais recente para o mais antigo."""
        if not self.directory.is_dir():
            return ()
        found = [
            self._read(folder.name)
            for folder in self.directory.iterdir()
            if _SAFE_ID.fullmatch(folder.name) and (folder / _LIBRARY_FILE).is_file()
        ]
        return tuple(sorted(found, key=lambda entry: entry.added_at, reverse=True))

    def entry(self, package_id: str) -> LibraryEntry:
        """Um pacote da biblioteca.

        Raises:
            PackageNotFoundError: Se não houver nenhum com este identificador.
        """
        if not self._exists(package_id):
            raise PackageNotFoundError(package_id)
        return self._read(package_id)

    def find(self, sha256: str) -> LibraryEntry | None:
        """O pacote com este conteúdo, se a biblioteca já o tiver."""
        return next((entry for entry in self.entries() if entry.sha256 == sha256), None)

    def data(self, package_id: str) -> bytes:
        """O ficheiro ``.mllab`` de um pacote.

        Raises:
            PackageNotFoundError: Se não houver nenhum com este identificador.
        """
        self.entry(package_id)
        return (self._folder(package_id) / _PACKAGE_FILE).read_bytes()

    def update(
        self, package_id: str, *, trusted: bool | None = None, run_id: str | None = None
    ) -> LibraryEntry:
        """Muda o que se sabe de um pacote (se já se confia nele, o run que pôs no histórico).

        Raises:
            PackageNotFoundError: Se não houver nenhum com este identificador.
        """
        entry = self.entry(package_id)
        changed = replace(
            entry,
            trusted=entry.trusted if trusted is None else trusted,
            run_id=entry.run_id if run_id is None else run_id,
        )
        self._write(changed)
        return changed

    def delete(self, package_id: str) -> None:
        """Apaga um pacote da biblioteca (não faz nada se não houver)."""
        if _SAFE_ID.fullmatch(package_id):
            shutil.rmtree(self._folder(package_id), ignore_errors=True)

    def _exists(self, package_id: str) -> bool:
        return (
            bool(_SAFE_ID.fullmatch(package_id))
            and (self._folder(package_id) / _LIBRARY_FILE).is_file()
        )

    def _folder(self, package_id: str) -> Path:
        # O identificador vira o nome de uma pasta: só um nome do formato certo chega aqui, e
        # por isso nunca se escreve (nem apaga) fora da biblioteca.
        if not _SAFE_ID.fullmatch(package_id):
            raise PackageNotFoundError(package_id)
        return self.directory / package_id

    def _write(self, entry: LibraryEntry) -> None:
        record = {
            "file_name": entry.file_name,
            "sha256": entry.sha256,
            "size": entry.size,
            "added_at": entry.added_at.isoformat(),
            "origin": entry.origin,
            "trusted": entry.trusted,
            "run_id": entry.run_id,
            "made_here": entry.made_here,
            "intact": entry.intact,
            "problems": list(entry.problems),
            "manifest": dict(entry.manifest),
        }
        text = json.dumps(record, ensure_ascii=False, indent=2)
        (self._folder(entry.package_id) / _LIBRARY_FILE).write_text(text, encoding="utf-8")

    def _read(self, package_id: str) -> LibraryEntry:
        path = self._folder(package_id) / _LIBRARY_FILE
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            return LibraryEntry(
                package_id=package_id,
                file_name=str(record["file_name"]),
                sha256=str(record["sha256"]),
                size=int(record["size"]),
                added_at=datetime.fromisoformat(record["added_at"]),
                origin="made_here" if record["origin"] == "made_here" else "loaded",
                trusted=bool(record["trusted"]),
                run_id=record.get("run_id"),
                made_here=bool(record.get("made_here", False)),
                intact=bool(record.get("intact", True)),
                problems=tuple(str(problem) for problem in record.get("problems", [])),
                manifest=dict(record.get("manifest", {})),
            )
        except (OSError, ValueError, KeyError) as error:
            raise TrackingError(
                f"The library's package '{package_id}' is damaged: {error}."
            ) from error


def new_package_id() -> str:
    """Um identificador novo de pacote (``pkg-`` e oito algarismos hexadecimais)."""
    return f"pkg-{secrets.token_hex(4)}"


def _ordered(files: Sequence[PackageFile]) -> tuple[PackageFile, ...]:
    return tuple(sorted(files, key=lambda file: FILE_ORDER.index(file.path)))


def _manifest_of(contents: PackageContents) -> dict[str, Any]:
    """O manifesto de um pacote lido, confirmado (o formato e a versão).

    Raises:
        PackageError: Se faltar, não for do formato, ou for de uma versão mais nova.
    """
    if contents.get(MANIFEST) is None:
        raise PackageError(
            "This ZIP isn't an Orbyte package: it lacks the manifest.json, which says what the "
            "package holds."
        )
    manifest = contents.manifest
    if manifest.get("format") != FORMAT:
        raise PackageError("This ZIP isn't an Orbyte package: the manifest is of another format.")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise PackageError(
            f"This package is from version {version} of the format, and this Orbyte only reads "
            f"up to {FORMAT_VERSION}: update Orbyte to open it."
        )
    return manifest


def _check(contents: PackageContents, manifest: Mapping[str, Any]) -> list[str]:
    """O que não confere entre os ficheiros e o manifesto (vazio num pacote intacto)."""
    problems: list[str] = []
    listed = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    for name in REQUIRED:
        if contents.get(name) is None:
            problems.append(f"The file {name} is missing.")
    for file in contents.files:
        if file.path in (MANIFEST, SIGNATURE):
            continue
        item = listed.get(file.path)
        if item is None:
            problems.append(f"The file {file.path} isn't in the manifest.")
        elif item.get("sha256") != file.sha256 or item.get("bytes") != len(file.content):
            problems.append(f"The file {file.path} isn't what the manifest describes.")
    for name in listed:
        if contents.get(name) is None:
            problems.append(f"The manifest describes {name}, and the package doesn't have it.")
    return problems


def _file_name(name: str) -> str:
    """Um nome de ficheiro seguro para descarregar (sem pastas), com a extensão do formato."""
    base = Path(name).name.strip() or "model"
    return base if base.endswith(EXTENSION) else f"{base}{EXTENSION}"


def _zip_time(created_at: object) -> tuple[int, int, int, int, int, int]:
    """A data dos ficheiros dentro do ZIP: a do pacote (o ZIP não aceita antes de 1980)."""
    try:
        moment = datetime.fromisoformat(str(created_at))
    except ValueError:
        moment = datetime.now(UTC)
    return (
        max(moment.year, 1980),
        moment.month,
        moment.day,
        moment.hour,
        moment.minute,
        moment.second,
    )
