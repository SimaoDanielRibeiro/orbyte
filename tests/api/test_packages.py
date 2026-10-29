"""Os pacotes pela API (ecrã Guardar): pré-visualizar, guardar, descarregar e carregar."""

import io
import zipfile
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

type Waiter = Callable[[str], dict[str, Any]]


def trained(client: TestClient, finished: Waiter, config: Callable[..., dict[str, Any]]) -> str:
    run_id: str = client.post("/experiments", json=config("logistic_regression")).json()["run_id"]
    assert finished(run_id)["status"] == "finished"
    return run_id


def test_the_package_is_previewed_saved_and_downloaded(
    client: TestClient, finished: Waiter, config: Callable[..., dict[str, Any]]
) -> None:
    run_id = trained(client, finished, config)

    candidates = client.get("/packages/candidates").json()
    options = client.get(f"/packages/runs/{run_id}/options").json()
    preview = client.post(f"/packages/runs/{run_id}/preview", json={"examples": 2}).json()
    saved = client.post(f"/packages/runs/{run_id}", json={"name": "Íris"})
    download = client.get(f"/packages/{saved.json()['package_id']}/file")

    assert [candidate["run_id"] for candidate in candidates] == [run_id]
    assert options["file_name"] == "iris-regressao-logistica.mllab"
    assert preview["parts"][0]["path"] == "README.md"
    assert preview["parts"][0]["content"].startswith("# Iris — Regressão logística")
    assert saved.status_code == 201
    assert saved.json()["file_name"] == "iris.mllab"
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert download.headers["content-disposition"] == 'attachment; filename="iris.mllab"'
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert "model.joblib" in archive.namelist()
    assert client.get("/packages").json()[0]["package_id"] == saved.json()["package_id"]


def test_a_package_is_loaded_opened_and_deleted(
    client: TestClient, finished: Waiter, config: Callable[..., dict[str, Any]]
) -> None:
    run_id = trained(client, finished, config)
    package_id = client.post(f"/packages/runs/{run_id}", json={}).json()["package_id"]
    data = client.get(f"/packages/{package_id}/file").content
    assert client.delete(f"/packages/{package_id}").status_code == 204

    loaded = client.post(
        "/packages/uploads", files={"file": ("iris.mllab", data, "application/zip")}
    )
    view = client.get(f"/packages/{loaded.json()['package_id']}").json()
    trusted = client.post(f"/packages/{loaded.json()['package_id']}/trust")

    assert loaded.status_code == 201
    assert loaded.json()["trusted"]
    assert loaded.json()["run_id"] == run_id
    assert view["package"]["origin"] == "loaded"
    assert [part["path"] for part in view["preview"]["parts"]][:3] == [
        "README.md",
        "manifest.json",
        "manifest.sig",
    ]
    assert trusted.status_code == 200


def test_what_is_not_a_package_is_refused_clearly(client: TestClient) -> None:
    response = client.post(
        "/packages/uploads", files={"file": ("dados.csv", b"a,b\n1,2\n", "text/csv")}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "package"
    assert "não é um pacote do ML Lab" in response.json()["message"]
    assert client.get("/packages/pkg-00000000").status_code == 404
