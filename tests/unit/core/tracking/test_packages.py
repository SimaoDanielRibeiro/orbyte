import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from ml_lab.core.foundation.errors import PackageError, PackageNotFoundError
from ml_lab.core.tracking.packages import (
    EXAMPLES,
    FORMAT,
    FORMAT_VERSION,
    INPUT,
    MANIFEST,
    MODEL,
    PREPARATION,
    README,
    RUN,
    SIGNATURE,
    PackageFile,
    PackageKey,
    PackageLibrary,
    assemble,
    load_model,
    model_bytes,
    read_package,
)

MANIFEST_BASE = {
    "format": FORMAT,
    "format_version": FORMAT_VERSION,
    "name": "Teste",
    "created_at": "2026-10-29T14:32:05+00:00",
}


def parts(*, with_model: bool = True) -> list[PackageFile]:
    model = LogisticRegression().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    files = [
        PackageFile(README, b"# Teste\n"),
        PackageFile(RUN, b'{"run_id": "run-1"}'),
        PackageFile(INPUT, b'{"columns": []}'),
        PackageFile(PREPARATION, b'{"transforms": []}'),
    ]
    if with_model:
        files.append(PackageFile(MODEL, model_bytes(model)))
    return files


@pytest.fixture
def key(tmp_path: Path) -> PackageKey:
    return PackageKey(tmp_path / "package.key")


def test_the_manifest_lists_every_file_and_is_signed(key: PackageKey) -> None:
    contents = assemble(parts(), MANIFEST_BASE, key)

    names = [file.path for file in contents.files]
    assert names == [README, MANIFEST, SIGNATURE, INPUT, PREPARATION, RUN, MODEL]
    listed = {item["path"]: item for item in contents.manifest["files"]}
    assert set(listed) == {README, INPUT, PREPARATION, RUN, MODEL}
    assert listed[README]["bytes"] == len(b"# Teste\n")
    assert key.verify(contents.get(MANIFEST) or b"", contents.text(SIGNATURE) or "")


def test_the_zip_has_exactly_the_assembled_files(key: PackageKey) -> None:
    contents = assemble(parts(), MANIFEST_BASE, key)

    with zipfile.ZipFile(io.BytesIO(contents.to_zip())) as archive:
        inside = {name: archive.read(name) for name in archive.namelist()}

    assert inside == {file.path: file.content for file in contents.files}


def test_a_package_made_here_opens_with_its_model(key: PackageKey) -> None:
    data = assemble(parts(), MANIFEST_BASE, key).to_zip()

    opened = read_package(data, key)

    assert opened.made_here
    assert opened.intact
    assert opened.problems == ()
    assert load_model(opened).predict([[3.0]]).tolist() == [1]


def test_a_package_from_another_installation_reads_but_is_not_made_here(
    key: PackageKey, tmp_path: Path
) -> None:
    other = PackageKey(tmp_path / "other.key")
    data = assemble(parts(), MANIFEST_BASE, other).to_zip()

    opened = read_package(data, key)

    assert not opened.made_here
    assert opened.intact
    assert opened.manifest["name"] == "Teste"


def test_a_changed_file_is_caught_and_the_model_never_opens(key: PackageKey) -> None:
    contents = assemble(parts(), MANIFEST_BASE, key)
    changed = [
        PackageFile(file.path, b'{"run_id": "run-2"}') if file.path == RUN else file
        for file in contents.files
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for file in changed:
            archive.writestr(file.path, file.content)

    opened = read_package(buffer.getvalue(), key)

    assert not opened.intact
    assert not opened.made_here
    assert "run.json não é o que o manifesto descreve" in opened.problems[0]
    with pytest.raises(PackageError, match="estragado ou foi alterado"):
        load_model(opened)


def test_a_file_whose_bytes_broke_is_reported_and_the_rest_reads(key: PackageKey) -> None:
    contents = assemble(parts(), MANIFEST_BASE, key)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in contents.files:
            archive.writestr(file.path, file.content)
    # Uma letra trocada dentro do input.json: o ZIP já não confere com a sua própria impressão.
    data = buffer.getvalue().replace(b'{"columns": []}', b'{"columns": [}]', 1)

    opened = read_package(data, key)

    assert not opened.intact
    assert not opened.made_here
    assert "input.json está estragado e não se lê" in opened.problems[0]
    assert opened.contents.get(README) == b"# Teste\n"


def test_files_the_format_does_not_know_are_reported(key: PackageKey) -> None:
    data = assemble(parts(), MANIFEST_BASE, key).to_zip()
    buffer = io.BytesIO(data)
    with zipfile.ZipFile(buffer, "a") as archive:
        archive.writestr("../../fora.txt", b"nada")

    opened = read_package(buffer.getvalue(), key)

    assert not opened.intact
    assert any("não conhece: ../../fora.txt" in problem for problem in opened.problems)


def test_what_is_not_a_package_is_explained(key: PackageKey) -> None:
    with pytest.raises(PackageError, match="não abre como um"):
        read_package(b"isto nao e um zip", key)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(README, b"# Sem manifesto")
    with pytest.raises(PackageError, match=r"falta o manifest\.json"):
        read_package(buffer.getvalue(), key)


def test_a_newer_format_asks_for_a_newer_lab(key: PackageKey) -> None:
    newer = {**MANIFEST_BASE, "format_version": FORMAT_VERSION + 1}
    data = assemble(parts(), newer, key).to_zip()

    with pytest.raises(PackageError, match=f"versão {FORMAT_VERSION + 1} do formato"):
        read_package(data, key)


def test_a_package_without_the_model_says_so(key: PackageKey) -> None:
    opened = read_package(assemble(parts(with_model=False), MANIFEST_BASE, key).to_zip(), key)

    with pytest.raises(PackageError, match="não traz o modelo"):
        load_model(opened)


def test_only_the_names_of_the_format_can_be_assembled(key: PackageKey) -> None:
    with pytest.raises(PackageError, match="nomes do formato"):
        assemble([*parts(), PackageFile("outro.txt", b"")], MANIFEST_BASE, key)
    with pytest.raises(PackageError, match="nomes do formato"):
        assemble([PackageFile(EXAMPLES, b"")], MANIFEST_BASE, key)


def test_the_key_is_created_once_and_kept_private(key: PackageKey) -> None:
    first = key.sign(b"dados")

    assert key.path.is_file()
    if sys.platform != "win32":
        # O Windows não tem as permissões POSIX: lá, o chmod só liga ou desliga o "só de
        # leitura", e o modo aparece sempre como 0o666 (a pasta do utilizador já é privada).
        assert key.path.stat().st_mode & 0o777 == 0o600
    assert key.sign(b"dados") == first
    assert not PackageKey(key.path.parent / "nenhuma.key").verify(b"dados", first)


def test_the_library_keeps_the_file_and_what_is_known_about_it(
    key: PackageKey, tmp_path: Path
) -> None:
    library = PackageLibrary(tmp_path / "packages")
    data = assemble(parts(), MANIFEST_BASE, key).to_zip()

    entry = library.add(data, file_name="titanic", origin="loaded", trusted=False)
    trusted = library.update(entry.package_id, trusted=True, run_id="run-1")

    assert entry.file_name == "titanic.mllab"
    assert library.data(entry.package_id) == data
    assert library.find(entry.sha256) == trusted
    assert library.entries() == (trusted,)
    assert (trusted.trusted, trusted.run_id) == (True, "run-1")
    library.delete(entry.package_id)
    assert library.entries() == ()
    with pytest.raises(PackageNotFoundError):
        library.entry(entry.package_id)


def test_the_library_never_leaves_its_folder(tmp_path: Path) -> None:
    library = PackageLibrary(tmp_path / "packages")

    entry = library.add(b"zip", file_name="../../fora.mllab", origin="loaded", trusted=False)

    assert entry.file_name == "fora.mllab"
    with pytest.raises(PackageNotFoundError):
        library.entry("../../etc")
    library.delete("../../etc")  # não faz nada
    assert (
        json.loads((tmp_path / "packages" / entry.package_id / "library.json").read_text())[
            "origin"
        ]
        == "loaded"
    )
