"""Contrato DataSource: obter dados de uma origem.

Uma fonte de dados sabe ler uma origem (um ficheiro, um dataset de exemplo, um
gerador de dados sintéticos) e devolve a tabela tal como chegou: um DataFrame e os
seus metadados (``LoadedTable``). O ``Dataset`` com arrays NumPy só existe no fim
da camada de dados, depois da limpeza, da divisão e das transformações (ADR 0009).

Cada tipo de fonte regista-se em ``DATA_SOURCES`` e descreve-se com um
``SourceSpec`` (tipo + parâmetros), que é o que fica guardado com a experiência:
com o mesmo ``SourceSpec`` obtêm-se os mesmos dados.

Para acrescentar um tipo de fonte (por exemplo, ler de uma URL): cria um ficheiro
nesta pasta com uma subclasse de ``DataSource`` decorada com
``@DATA_SOURCES.register("url")``.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.registry import Registry


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Referência reproduzível a uma origem de dados.

    Attributes:
        kind: Tipo de fonte, igual ao nome com que se registou (``"sample"``, ``"file"``...).
        params: Parâmetros da fonte, só com valores simples (texto, números, listas).
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Identificador curto e estável: a mesma origem dá sempre a mesma chave."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """A referência como dicionário simples, para JSON."""
        return {"kind": self.kind, "params": dict(self.params)}


@dataclass(frozen=True, slots=True, eq=False)
class LoadedTable:
    """A tabela tal como chegou da origem, com os seus metadados.

    Attributes:
        name: Nome a mostrar, por exemplo ``"Titanic"``.
        frame: Os dados. Nenhuma etapa o altera: cada uma devolve um DataFrame novo.
        spec: De onde vieram os dados, para os poder voltar a obter.
        description: Texto que explica o dataset.
        suggested_target: Coluna-alvo habitual deste dataset, se houver.
        notices: Avisos da leitura (codificação do ficheiro, folhas ignoradas...).
    """

    name: str
    frame: pd.DataFrame
    spec: SourceSpec
    description: str = ""
    suggested_target: str | None = None
    notices: tuple[Notice, ...] = ()


class DataSource(ABC):
    """Contrato de uma fonte de dados."""

    @abstractmethod
    def load(self) -> LoadedTable:
        """Lê a origem e devolve a tabela.

        Raises:
            DataSourceError: Se não for possível obter os dados.
            DataValidationError: Se os dados obtidos não forem utilizáveis.
        """

    @abstractmethod
    def spec(self) -> SourceSpec:
        """A referência que permite voltar a criar esta fonte com ``open_source``."""


DATA_SOURCES: Registry[Callable[..., DataSource]] = Registry(
    "data_sources", discover="ml_lab.core.data.io"
)
"""Tipos de fonte de dados, pelo nome: ``"file"``, ``"sample"`` e ``"synthetic"``."""


def open_source(spec: SourceSpec) -> DataSource:
    """Cria a fonte de dados descrita por ``spec``.

    Raises:
        ComponentNotFoundError: Se o tipo de fonte não existir.
        InvalidParameterError: Se os parâmetros não servirem para esse tipo de fonte.
    """
    factory = DATA_SOURCES.get(spec.kind)
    try:
        return factory(**spec.params)
    except TypeError as error:
        raise InvalidParameterError(
            f"Parâmetros inválidos para a fonte '{spec.kind}': {error}."
        ) from error
