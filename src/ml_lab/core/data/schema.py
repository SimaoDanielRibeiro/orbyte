"""Descrição das colunas: o tipo de cada uma e o seu papel (feature, alvo ou ignorar).

O tipo (``ColumnKind``) é o que a coluna *significa*, que nem sempre coincide com o
tipo de dados do pandas: um preço lido como texto continua a ser numérico, e um
inteiro que só vale 0 ou 1 é booleano. O tipo decide que transformação a coluna
recebe; o papel decide se entra no modelo.

Exemplo:
    >>> schema = TableSchema.of(
    ...     {
    ...         "Age": ColumnKind.NUMERIC,
    ...         "Sex": ColumnKind.CATEGORICAL,
    ...         "Survived": ColumnKind.BOOLEAN,
    ...     }
    ... ).with_target("Survived")
    >>> schema.target.name
    'Survived'
    >>> schema.feature_names()
    ('Age', 'Sex')
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self

from ml_lab.core.foundation.errors import ColumnNotFoundError, InvalidParameterError


class ColumnKind(StrEnum):
    """O que uma coluna significa, independentemente de como está guardada."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em português."""
        return _KIND_LABELS[self]

    @property
    def is_usable_by_models(self) -> bool:
        """Se a coluna pode entrar num modelo depois de transformada.

        Texto livre e identificadores não entram: um identificador é diferente em
        cada linha, por isso não ensina nada que sirva para dados novos.
        """
        return self not in (ColumnKind.TEXT, ColumnKind.IDENTIFIER)


class ColumnRole(StrEnum):
    """Papel de uma coluna na experiência."""

    FEATURE = "feature"
    TARGET = "target"
    IGNORED = "ignored"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em português."""
        return _ROLE_LABELS[self]


_KIND_LABELS: Final[Mapping[ColumnKind, str]] = MappingProxyType(
    {
        ColumnKind.NUMERIC: "Numérica",
        ColumnKind.CATEGORICAL: "Categórica",
        ColumnKind.BOOLEAN: "Booleana",
        ColumnKind.DATETIME: "Data",
        ColumnKind.TEXT: "Texto",
        ColumnKind.IDENTIFIER: "Identificador",
    }
)

_ROLE_LABELS: Final[Mapping[ColumnRole, str]] = MappingProxyType(
    {ColumnRole.FEATURE: "Feature", ColumnRole.TARGET: "Alvo", ColumnRole.IGNORED: "Ignorada"}
)


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """Uma coluna: nome, tipo e papel.

    Attributes:
        name: Nome da coluna, tal como está nos dados.
        kind: O que a coluna significa.
        role: Se é feature, alvo ou ignorada.
    """

    name: str
    kind: ColumnKind
    role: ColumnRole = ColumnRole.FEATURE


@dataclass(frozen=True, slots=True)
class TableSchema:
    """O esquema de uma tabela: as suas colunas, pela ordem dos dados.

    É imutável: os métodos ``with_*`` devolvem um esquema novo.

    Attributes:
        columns: As colunas, sem nomes repetidos e com no máximo um alvo.
    """

    columns: tuple[ColumnSchema, ...]

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise InvalidParameterError(f"Colunas repetidas no esquema: {', '.join(repeated)}.")
        targets = [column.name for column in self.columns if column.role is ColumnRole.TARGET]
        if len(targets) > 1:
            raise InvalidParameterError(
                f"Um esquema só pode ter um alvo; tem {len(targets)}: {', '.join(targets)}."
            )

    @classmethod
    def of(cls, kinds: Mapping[str, ColumnKind]) -> Self:
        """Cria um esquema em que todas as colunas são features."""
        return cls(tuple(ColumnSchema(name, kind) for name, kind in kinds.items()))

    @property
    def names(self) -> tuple[str, ...]:
        """Nomes de todas as colunas, pela ordem dos dados."""
        return tuple(column.name for column in self.columns)

    @property
    def target(self) -> ColumnSchema | None:
        """A coluna-alvo, se houver."""
        return next((c for c in self.columns if c.role is ColumnRole.TARGET), None)

    def feature_names(self, *kinds: ColumnKind) -> tuple[str, ...]:
        """Nomes das features; se forem dados tipos, só as desses tipos."""
        return tuple(
            column.name
            for column in self.columns
            if column.role is ColumnRole.FEATURE and (not kinds or column.kind in kinds)
        )

    def with_target(self, name: str | None) -> Self:
        """Esquema com ``name`` como alvo; o alvo anterior volta a ser feature.

        Com ``None``, o esquema fica sem alvo (aprendizagem não supervisionada).

        Raises:
            ColumnNotFoundError: Se não existir nenhuma coluna ``name``.
        """
        if name is not None:
            self._require(name)
        return replace(
            self,
            columns=tuple(replace(column, role=_role_for(column, name)) for column in self.columns),
        )

    def with_roles(self, roles: Mapping[str, ColumnRole]) -> Self:
        """Esquema com os papéis dados; as colunas que não aparecem ficam iguais.

        Raises:
            ColumnNotFoundError: Se alguma das colunas não existir.
        """
        self._require(*roles)
        updated = tuple(
            replace(column, role=roles.get(column.name, column.role)) for column in self.columns
        )
        return replace(self, columns=updated)

    def _require(self, *names: str) -> None:
        for name in names:
            if name not in self:
                raise ColumnNotFoundError(name, self.names)

    def __getitem__(self, name: str) -> ColumnSchema:
        for column in self.columns:
            if column.name == name:
                return column
        raise ColumnNotFoundError(name, self.names)

    def __contains__(self, name: object) -> bool:
        return any(column.name == name for column in self.columns)

    def __iter__(self) -> Iterator[ColumnSchema]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)


def _role_for(column: ColumnSchema, target: str | None) -> ColumnRole:
    if column.name == target:
        return ColumnRole.TARGET
    if column.role is ColumnRole.TARGET:
        return ColumnRole.FEATURE if column.kind.is_usable_by_models else ColumnRole.IGNORED
    return column.role
