"""Descarrega os datasets de exemplo para ``data/samples/``.

Os ficheiros já estão versionados no Git, por isso só precisas deste script se os
apagares ou quiseres confirmar que são iguais aos originais. Cada ficheiro é
verificado com SHA-256: se a origem mudar, o script pára em vez de gravar dados
diferentes dos que os testes esperam.

Uso, a partir da raiz do projeto::

    uv run python scripts/download_samples.py            # só os que faltam
    uv run python scripts/download_samples.py --force    # todos, por cima dos atuais
"""

import argparse
import gzip
import hashlib
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_lab.config import get_settings
from ml_lab.logging import configure_logging, get_logger

logger = get_logger("scripts.download_samples")

_TIMEOUT_SECONDS = 60

_WINE_HEADER = (
    "cultivar,alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,"
    "nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280/od315_of_diluted_wines,"
    "proline\n"
)


def _unchanged(content: bytes) -> bytes:
    return content


def _add_wine_header(content: bytes) -> bytes:
    # O ficheiro da UCI não tem cabeçalho; os nomes são os do scikit-learn.
    return _WINE_HEADER.encode("ascii") + content


def _compress(content: bytes) -> bytes:
    # mtime=0 torna o ficheiro comprimido sempre igual, byte a byte.
    return gzip.compress(content, compresslevel=9, mtime=0)


@dataclass(frozen=True, slots=True)
class Download:
    """Um dataset de exemplo: de onde vem, como se confirma e onde fica."""

    filename: str
    url: str
    sha256: str
    convert: Callable[[bytes], bytes] = _unchanged


DOWNLOADS = (
    # Titanic do Kaggle (train.csv, 891 passageiros), tal como no tutorial do pandas.
    Download(
        "titanic.csv",
        "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv",
        "cd888141e6a2ca5906114d0b6a63d8a3e14edb3326f7bdd56327d14646b68c36",
    ),
    # Iris de Fisher (1936), com os valores corrigidos do R (a cópia da UCI tem dois erros).
    Download(
        "iris.csv",
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "9cc1c345c71bcc9b486b74cbf6063fa66f4bb5e0f603a4b3c3471ec2e5e8e355",
    ),
    # Wine da UCI: análise química de vinhos de três castas italianas.
    Download(
        "wine.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659",
        _add_wine_header,
    ),
    # California Housing (censo de 1990), na versão do livro "Hands-On Machine Learning",
    # com valores em falta e uma coluna categórica. Comprimido: tem 20 640 linhas.
    Download(
        "california_housing.csv.gz",
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e",
        _compress,
    ),
)


def fetch(download: Download) -> bytes:
    """Descarrega o ficheiro e confirma o SHA-256.

    Raises:
        RuntimeError: Se o conteúdo não for o esperado.
    """
    with urllib.request.urlopen(download.url, timeout=_TIMEOUT_SECONDS) as response:
        content: bytes = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != download.sha256:
        raise RuntimeError(
            f"{download.filename}: o conteúdo descarregado de {download.url} mudou "
            f"(SHA-256 {digest}, esperado {download.sha256}). Confirma a origem antes de "
            "atualizar o valor esperado neste script."
        )
    return content


def main(argv: list[str] | None = None) -> int:
    """Descarrega os datasets em falta (ou todos, com ``--force``)."""
    parser = argparse.ArgumentParser(description="Descarrega os datasets de exemplo.")
    parser.add_argument("--force", action="store_true", help="descarrega também os que já existem")
    parser.add_argument("--output", type=Path, help="pasta de destino (por omissão, data/samples)")
    args = parser.parse_args(argv)

    configure_logging()
    output: Path = args.output or get_settings().samples_dir
    output.mkdir(parents=True, exist_ok=True)
    for download in DOWNLOADS:
        target = output / download.filename
        if target.exists() and not args.force:
            logger.info("já existe, não descarreguei", extra={"file": download.filename})
            continue
        target.write_bytes(download.convert(fetch(download)))
        logger.info("gravado", extra={"file": download.filename, "bytes": target.stat().st_size})
    return 0


if __name__ == "__main__":
    sys.exit(main())
