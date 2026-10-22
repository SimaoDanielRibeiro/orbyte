import re
from pathlib import Path

import pytest

from ml_lab.config import KNOWN_VARIABLES, Settings, get_settings, project_root
from ml_lab.core.foundation.errors import ConfigurationError


def write_env_file(home: Path, content: str) -> None:
    (home / ".env").write_text(content, encoding="utf-8")


def test_defaults_point_inside_the_project(tmp_path: Path) -> None:
    settings = Settings.from_env({}, home=tmp_path)

    assert settings.home == tmp_path.resolve()
    assert settings.samples_dir == settings.home / "data" / "samples"
    assert settings.uploads_dir == settings.home / "data" / "uploads"
    assert settings.artifacts_dir == settings.home / "artifacts"
    assert settings.database_path == settings.home / "artifacts" / "ml_lab.db"
    assert settings.log_level == "INFO"
    assert settings.log_format == "text"


def test_environment_variables_override_the_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {"ML_LAB_LOG_LEVEL": "debug", "ML_LAB_LOG_FORMAT": "JSON"}, home=tmp_path
    )

    assert settings.log_level == "DEBUG"
    assert settings.log_format == "json"


def test_relative_paths_are_resolved_from_the_project_root(tmp_path: Path) -> None:
    settings = Settings.from_env({"ML_LAB_UPLOADS_DIR": "meus_dados"}, home=tmp_path)

    assert settings.uploads_dir == settings.home / "meus_dados"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    elsewhere = tmp_path / "noutro_sitio"

    settings = Settings.from_env({"ML_LAB_ARTIFACTS_DIR": str(elsewhere)}, home=tmp_path)

    assert settings.artifacts_dir == elsewhere


def test_database_follows_a_custom_artifacts_dir(tmp_path: Path) -> None:
    settings = Settings.from_env({"ML_LAB_ARTIFACTS_DIR": "saidas"}, home=tmp_path)

    assert settings.database_path == settings.home / "saidas" / "ml_lab.db"


def test_env_file_is_read(tmp_path: Path) -> None:
    write_env_file(
        tmp_path,
        "# Comentários e linhas vazias são ignorados\n\n"
        "ML_LAB_LOG_LEVEL=warning\n"
        "export ML_LAB_LOG_FORMAT='json'\n",
    )

    settings = Settings.from_env({}, home=tmp_path)

    assert settings.log_level == "WARNING"
    assert settings.log_format == "json"


def test_env_file_saved_with_a_bom_is_read(tmp_path: Path) -> None:
    # O Bloco de Notas e o PowerShell 5.1 gravam UTF-8 com BOM no início do ficheiro.
    (tmp_path / ".env").write_text("ML_LAB_LOG_LEVEL=DEBUG\n", encoding="utf-8-sig")

    assert Settings.from_env({}, home=tmp_path).log_level == "DEBUG"


def test_environment_wins_over_the_env_file(tmp_path: Path) -> None:
    write_env_file(tmp_path, "ML_LAB_LOG_LEVEL=ERROR\n")

    settings = Settings.from_env({"ML_LAB_LOG_LEVEL": "DEBUG"}, home=tmp_path)

    assert settings.log_level == "DEBUG"


def test_variables_of_other_programs_are_ignored(tmp_path: Path) -> None:
    write_env_file(tmp_path, "STREAMLIT_SERVER_PORT=8502\n")

    Settings.from_env({"PATH": "/usr/bin", "HOME": "/home/simao"}, home=tmp_path)


def test_unknown_variable_is_rejected_with_a_suggestion(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Querias dizer ML_LAB_LOG_LEVEL"):
        Settings.from_env({"ML_LAB_LOGLEVEL": "DEBUG"}, home=tmp_path)


def test_invalid_log_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="ML_LAB_LOG_LEVEL"):
        Settings.from_env({"ML_LAB_LOG_LEVEL": "VERBOSE"}, home=tmp_path)


def test_invalid_log_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="ML_LAB_LOG_FORMAT"):
        Settings.from_env({"ML_LAB_LOG_FORMAT": "xml"}, home=tmp_path)


def test_malformed_env_file_line_points_to_the_line(tmp_path: Path) -> None:
    write_env_file(tmp_path, "ML_LAB_LOG_LEVEL=INFO\nesta linha não tem igual\n")

    with pytest.raises(ConfigurationError, match=r"\.env:2"):
        Settings.from_env({}, home=tmp_path)


def test_ensure_directories_creates_the_missing_folders(tmp_path: Path) -> None:
    settings = Settings.from_env({"ML_LAB_DATABASE_PATH": "db/lab.db"}, home=tmp_path)

    settings.ensure_directories()

    assert settings.uploads_dir.is_dir()
    assert settings.artifacts_dir.is_dir()
    assert (settings.home / "db").is_dir()


def test_project_root_is_the_folder_with_pyproject() -> None:
    assert (project_root() / "pyproject.toml").is_file()


def test_env_example_documents_every_known_variable() -> None:
    example = (project_root() / ".env.example").read_text(encoding="utf-8")

    assert set(re.findall(r"\bML_LAB_[A-Z_]+\b", example)) == KNOWN_VARIABLES


def test_get_settings_reads_the_environment_once() -> None:
    assert get_settings() is get_settings()
