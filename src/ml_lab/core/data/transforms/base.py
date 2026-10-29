"""Contrato Transform: aprende com os dados de treino e transforma quaisquer dados.

Uma transformação tem dois momentos, como no scikit-learn:

- ``fit(treino)``: aprende os parâmetros (a mediana para preencher nulos, a média e
  o desvio para escalar, as categorias para codificar) **só com o treino**;
- ``transform(dados)``: aplica o que aprendeu, sem voltar a aprender, ao treino, à
  validação e ao teste.

Se aprendesse também com o teste, o modelo estaria a "espreitar" as respostas: é o
*data leakage*. Separar os dois momentos é o que o impede.

As transformações nunca alteram a tabela recebida: devolvem sempre uma nova.

O que uma transformação aprendeu sai em valores simples (``learned``), e com eles e a
configuração (``params``) ela volta a existir, já treinada (``restore``). É assim que a
preparação viaja com o modelo num pacote, em JSON e sem pickle (ADR 0039).
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, override

import pandas as pd

from ml_lab.core.data.lineage import LineageStep, Stage
from ml_lab.core.foundation.errors import ColumnNotFoundError, InvalidParameterError, NotFittedError

_BY_NAME: dict[str, type["Transform"]] = {}
"""Cada transformação pelo seu ``name`` (preenchido ao definir a classe)."""


class Transform(ABC):
    """Uma transformação que aprende com o treino.

    As subclasses implementam ``_fit`` (aprender), ``_transform`` (aplicar) e
    ``describe`` (explicar o que aprenderam).

    Args:
        columns: As colunas em que a transformação atua.

    Attributes:
        name: Nome estável da operação, usado na linhagem.
        title: Nome a mostrar, em português.
    """

    name: ClassVar[str]
    title: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Uma transformação nova fica logo conhecida pelo nome: é o que o restore precisa.
        if "name" in cls.__dict__:
            _BY_NAME[cls.name] = cls

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self._fitted = False
        self._columns_before: tuple[str, ...] = ()
        self._columns_after: tuple[str, ...] = ()
        self._rows_fitted = 0

    @property
    def is_fitted(self) -> bool:
        """Se a transformação já aprendeu com os dados de treino."""
        return self._fitted

    def fit(self, frame: pd.DataFrame) -> Self:
        """Aprende os parâmetros com ``frame`` (os dados de treino)."""
        self._require(frame)
        self._fit(frame)
        self._fitted = True
        self._rows_fitted = len(frame)
        self._columns_before = tuple(str(name) for name in frame.columns)
        self._columns_after = tuple(str(name) for name in self._transform(frame.iloc[:0]).columns)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aplica o que aprendeu a ``frame``, devolvendo uma tabela nova.

        Raises:
            NotFittedError: Se ainda não tiver aprendido (``fit``).
            ColumnNotFoundError: Se faltar alguma das colunas.
        """
        if not self._fitted:
            raise NotFittedError(
                f"A transformação '{self.title}' ainda não aprendeu com o treino: chama fit "
                "com os dados de treino antes de transform."
            )
        self._require(frame)
        return self._transform(frame)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aprende com ``frame`` e transforma-o."""
        return self.fit(frame).transform(frame)

    def params(self) -> dict[str, Any]:
        """A configuração da transformação (o que é preciso para a repetir).

        São os argumentos do construtor, pelo nome: ``type(t)(**t.params())`` cria a mesma
        transformação, ainda por treinar.
        """
        return {"columns": list(self.columns)}

    def learned(self) -> dict[str, Any]:
        """O que a transformação aprendeu com o treino, em valores simples (vão para JSON).

        Raises:
            NotFittedError: Se ainda não tiver aprendido.
        """
        if not self._fitted:
            raise NotFittedError(f"A transformação '{self.title}' ainda não aprendeu.")
        return {
            **self._learned(),
            "rows_fitted": self._rows_fitted,
            "columns_before": list(self._columns_before),
            "columns_after": list(self._columns_after),
        }

    @classmethod
    def restore(cls, params: Mapping[str, Any], learned: Mapping[str, Any]) -> Self:
        """Volta a criar a transformação, já treinada, sem voltar a ver os dados de treino.

        Args:
            params: A configuração (o que ``params()`` devolveu).
            learned: O que aprendeu (o que ``learned()`` devolveu).

        Raises:
            InvalidParameterError: Se a configuração ou o que aprendeu estiverem incompletos.
        """
        try:
            transform = cls(**params)
            transform._restore(learned)
            transform._rows_fitted = int(learned["rows_fitted"])
            transform._columns_before = tuple(str(name) for name in learned["columns_before"])
            transform._columns_after = tuple(str(name) for name in learned["columns_after"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidParameterError(
                f"Não foi possível restaurar a transformação '{cls.title}': {error}."
            ) from error
        transform._fitted = True
        return transform

    @abstractmethod
    def describe(self) -> str:
        """O que a transformação faz e o que aprendeu com o treino, em português."""

    def lineage_step(self) -> LineageStep:
        """O passo de linhagem, com as colunas antes e depois (só depois do ``fit``).

        Raises:
            NotFittedError: Se ainda não tiver aprendido.
        """
        if not self._fitted:
            raise NotFittedError(f"A transformação '{self.title}' ainda não aprendeu.")
        before, after = self._columns_before, self._columns_after
        unchanged = set(before) & set(after)
        return LineageStep(
            stage=Stage.TRANSFORM,
            operation=self.name,
            description=self.describe(),
            params=self.params(),
            rows_before=self._rows_fitted,
            rows_after=self._rows_fitted,
            columns_added=tuple(c for c in after if c not in before),
            columns_removed=tuple(c for c in before if c not in after),
            columns_changed=tuple(c for c in self.columns if c in unchanged),
        )

    @abstractmethod
    def _fit(self, frame: pd.DataFrame) -> None:
        """Aprende com o treino; as colunas já foram confirmadas."""

    def _learned(self) -> dict[str, Any]:
        """O que a subclasse aprendeu, em valores simples (por omissão, nada)."""
        return {}

    def _restore(self, learned: Mapping[str, Any]) -> None:  # noqa: B027
        """Repõe o que a subclasse aprendeu, a partir de ``_learned`` (por omissão, nada)."""

    @abstractmethod
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aplica o que aprendeu; tem de funcionar também com uma tabela sem linhas."""

    def _require(self, frame: pd.DataFrame) -> None:
        available = [str(name) for name in frame.columns]
        for column in self.columns:
            if column not in frame.columns:
                raise ColumnNotFoundError(column, available)


class DropColumns(Transform):
    """Retira colunas que não podem entrar no modelo (não aprende nada).

    Args:
        columns: As colunas a retirar.
        reason: Porque são retiradas, para a linhagem.
    """

    name = "drop_columns"
    title = "Retirar colunas"

    def __init__(self, columns: Sequence[str], reason: str) -> None:
        super().__init__(columns)
        self.reason = reason

    @override
    def params(self) -> dict[str, Any]:
        return {**super().params(), "reason": self.reason}

    @override
    def describe(self) -> str:
        return f"Retiradas {', '.join(self.columns)}: {self.reason}"

    @override
    def _fit(self, frame: pd.DataFrame) -> None:
        pass

    @override
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=list(self.columns))


def transform_class(name: str) -> type[Transform]:
    """A transformação com este ``name`` (``"impute"``, ``"one_hot"``…).

    Raises:
        InvalidParameterError: Se não houver nenhuma com esse nome.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(_BY_NAME))
        raise InvalidParameterError(
            f"Transformação desconhecida: '{name}'. As que existem: {known}."
        ) from None


def transform_names() -> tuple[str, ...]:
    """O nome de todas as transformações conhecidas, por ordem alfabética."""
    return tuple(sorted(_BY_NAME))
