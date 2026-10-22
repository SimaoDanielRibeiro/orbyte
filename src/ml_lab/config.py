"""Definições do ML Lab: caminhos, base de dados e logs.

Usa só a biblioteca padrão. Cada valor vem do primeiro sítio onde estiver definido:

1. variáveis de ambiente (por exemplo ``ML_LAB_LOG_LEVEL=DEBUG``);
2. ficheiro ``.env`` na raiz do projeto (ver ``.env.example``);
3. valores por omissão.

Os caminhos relativos são resolvidos a partir da raiz do projeto. Uma variável
``ML_LAB_*`` desconhecida é um erro, e não é ignorada: um erro de escrita no nome
de uma variável não pode passar despercebido.
"""

import difflib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, Self

from ml_lab.core.foundation.errors import ConfigurationError
from ml_lab.logging import LOG_FORMATS, LOG_LEVELS, LogFormat

ENV_PREFIX: Final = "ML_LAB_"

UPLOADS_DIR_VAR: Final = "ML_LAB_UPLOADS_DIR"
ARTIFACTS_DIR_VAR: Final = "ML_LAB_ARTIFACTS_DIR"
DATABASE_PATH_VAR: Final = "ML_LAB_DATABASE_PATH"
LOG_LEVEL_VAR: Final = "ML_LAB_LOG_LEVEL"
LOG_FORMAT_VAR: Final = "ML_LAB_LOG_FORMAT"

KNOWN_VARIABLES: Final = frozenset(
    {UPLOADS_DIR_VAR, ARTIFACTS_DIR_VAR, DATABASE_PATH_VAR, LOG_LEVEL_VAR, LOG_FORMAT_VAR}
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Definições resolvidas do ML Lab. São imutáveis.

    Attributes:
        home: Raiz do projeto; os caminhos relativos partem daqui.
        samples_dir: Datasets de exemplo pequenos, versionados no Git.
        uploads_dir: Datasets carregados pelo utilizador (ignorados pelo Git).
        artifacts_dir: Modelos treinados e snapshots (ignorados pelo Git).
        database_path: Ficheiro SQLite do registo de experiências.
        log_level: Nível mínimo dos logs: DEBUG, INFO, WARNING, ERROR ou CRITICAL.
        log_format: ``"text"`` para ler no terminal, ``"json"`` para logs estruturados.
    """

    home: Path
    samples_dir: Path
    uploads_dir: Path
    artifacts_dir: Path
    database_path: Path
    log_level: str = "INFO"
    log_format: LogFormat = "text"

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, home: Path | None = None
    ) -> Self:
        """Lê as definições das variáveis de ambiente e do ficheiro ``.env``.

        Args:
            environ: Variáveis de ambiente a usar; por omissão, ``os.environ``.
            home: Raiz do projeto; por omissão, a pasta com o ``pyproject.toml``.

        Returns:
            As definições, já validadas.

        Raises:
            ConfigurationError: Se o ``.env`` estiver mal escrito ou se alguma variável
                for desconhecida ou tiver um valor inválido.
        """
        root = (home if home is not None else project_root()).resolve()
        values = {
            **_with_prefix(_read_env_file(root / ".env")),
            **_with_prefix(os.environ if environ is None else environ),
        }
        _reject_unknown(values)

        artifacts_dir = _path(values, ARTIFACTS_DIR_VAR, root, default="artifacts")
        database_path = _path(values, DATABASE_PATH_VAR, root, default=artifacts_dir / "ml_lab.db")
        return cls(
            home=root,
            samples_dir=root / "data" / "samples",
            uploads_dir=_path(values, UPLOADS_DIR_VAR, root, default="data/uploads"),
            artifacts_dir=artifacts_dir,
            database_path=database_path,
            log_level=_log_level(values.get(LOG_LEVEL_VAR, "INFO")),
            log_format=_log_format(values.get(LOG_FORMAT_VAR, "text")),
        )

    def ensure_directories(self) -> None:
        """Cria as pastas de trabalho que ainda não existam."""
        for directory in (
            self.samples_dir,
            self.uploads_dir,
            self.artifacts_dir,
            self.database_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@cache
def get_settings() -> Settings:
    """Definições do processo atual, lidas uma única vez.

    Nos testes, usa ``Settings.from_env(...)`` com valores explícitos. Para voltar
    a ler o ambiente, chama ``get_settings.cache_clear()``.
    """
    return Settings.from_env()


def project_root() -> Path:
    """Raiz do projeto: a pasta com o ``pyproject.toml``, duas pastas acima do pacote.

    Se o pacote estiver instalado fora do repositório, usa a pasta atual.
    """
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


def _read_env_file(path: Path) -> dict[str, str]:
    """Lê um ficheiro ``.env`` simples: linhas ``CHAVE=valor`` e comentários com ``#``."""
    if not path.is_file():
        return {}
    # utf-8-sig ignora o BOM que alguns editores do Windows põem no início do ficheiro;
    # sem isto, a primeira variável ficava com um carácter invisível e seria ignorada.
    text = path.read_text(encoding="utf-8-sig")
    values: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if not separator or not key.strip():
            raise ConfigurationError(
                f"{path}:{number}: linha inválida '{raw_line}'. Usa o formato CHAVE=valor."
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _with_prefix(variables: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in variables.items() if key.startswith(ENV_PREFIX)}


def _reject_unknown(values: Mapping[str, str]) -> None:
    unknown = sorted(set(values) - KNOWN_VARIABLES)
    if not unknown:
        return
    message = f"Variáveis de configuração desconhecidas: {', '.join(unknown)}."
    close = difflib.get_close_matches(unknown[0], KNOWN_VARIABLES, n=1)
    if close:
        message += f" Querias dizer {close[0]}?"
    raise ConfigurationError(f"{message} As variáveis aceites estão em .env.example.")


def _path(values: Mapping[str, str], variable: str, root: Path, *, default: str | Path) -> Path:
    raw = values.get(variable, "").strip()
    path = Path(raw).expanduser() if raw else Path(default)
    return path if path.is_absolute() else root / path


def _log_level(value: str) -> str:
    level = value.strip().upper()
    if level not in LOG_LEVELS:
        raise ConfigurationError(
            f"{LOG_LEVEL_VAR} inválido: '{value}'. Usa um destes: {', '.join(LOG_LEVELS)}."
        )
    return level


def _log_format(value: str) -> LogFormat:
    normalized = value.strip().lower()
    for log_format in LOG_FORMATS:
        if log_format == normalized:
            return log_format
    raise ConfigurationError(
        f"{LOG_FORMAT_VAR} inválido: '{value}'. Usa um destes: {', '.join(LOG_FORMATS)}."
    )
