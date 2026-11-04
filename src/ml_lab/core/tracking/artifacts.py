"""Os ficheiros de cada run guardado: o modelo treinado e o treino passo a passo.

Ficam numa pasta por run (``artifacts/runs/<run_id>/``):

- ``model.joblib``: o modelo treinado, gravado com o joblib (vem com o scikit-learn e
  sabe gravar matrizes do NumPy de forma eficiente). Um ficheiro joblib é um *pickle*:
  ao abri-lo, o Python volta a criar os objetos que lá estão, e por isso só se abrem
  ficheiros que o próprio ML Lab gravou nesta pasta, nunca ficheiros de fora;
- um ficheiro JSON por documento, com o nome do documento (``animation.json``,
  ``network.json``): os snapshots do treino. Pesam mais do que o resto do resultado (até
  alguns MB), por isso só se guardam a pedido (ADR 0020). Quem guarda decide quais são;
  aqui só se gravam e leem.
"""

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import joblib

from ml_lab.core.foundation.errors import TrackingError

MODEL_FILE: Final = "model.joblib"

_SAFE_NAME: Final = re.compile(r"[A-Za-z0-9_-]+")


class ArtifactStore:
    """A pasta dos ficheiros dos runs guardados.

    Args:
        directory: A pasta-mãe (normalmente ``artifacts/runs``); é criada quando faz falta.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, run_id: str, model: Any, documents: Mapping[str, Any] | None = None) -> None:
        """Grava o modelo treinado e os documentos do run.

        Args:
            run_id: O run.
            model: O modelo treinado.
            documents: Valores de JSON, pelo nome: ``{"animation": ...}`` fica em
                ``animation.json``.
        """
        folder = self._folder(run_id)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / MODEL_FILE)
        for name, document in (documents or {}).items():
            text = json.dumps(document, allow_nan=False)
            (folder / _document_file(name)).write_text(text, encoding="utf-8")

    def has_model(self, run_id: str) -> bool:
        """Se o run tem o modelo gravado."""
        return (self._folder(run_id) / MODEL_FILE).is_file()

    def model(self, run_id: str) -> Any:
        """O modelo treinado de um run guardado.

        Raises:
            TrackingError: Se o run não tiver o modelo gravado.
        """
        path = self._folder(run_id) / MODEL_FILE
        if not path.is_file():
            raise TrackingError(f"Run '{run_id}' has no saved model.")
        try:
            return joblib.load(path)
        except Exception as error:  # um pickle de outra versão do código falha de muitas formas
            raise TrackingError(f"The model of run '{run_id}' didn't open: {error}") from error

    def document(self, run_id: str, name: str) -> Any | None:
        """Um documento de um run guardado, em valores de JSON (``None`` se não o tiver)."""
        path = self._folder(run_id) / _document_file(name)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def delete(self, run_id: str) -> None:
        """Apaga os ficheiros de um run (não faz nada se não houver)."""
        shutil.rmtree(self._folder(run_id), ignore_errors=True)

    def _folder(self, run_id: str) -> Path:
        # O identificador vira o nome de uma pasta: um nome simples garante que nunca se
        # escreve (nem apaga) fora da pasta dos artefactos.
        if not _SAFE_NAME.fullmatch(run_id):
            raise TrackingError(f"Invalid run id: '{run_id}'.")
        return self.directory / run_id


def _document_file(name: str) -> str:
    # Pela mesma razão, o nome de um documento também tem de ser simples.
    if not _SAFE_NAME.fullmatch(name):
        raise TrackingError(f"Invalid document name: '{name}'.")
    return f"{name}.json"
