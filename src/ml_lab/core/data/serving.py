"""A preparação que viaja com o modelo: o que entra, como se limpa e como se transforma.

Depois de preparar os dados, o pipeline (com o que aprendeu no treino) e o perfil das
colunas de entrada ficam juntos numa ``FittedPreparation``. É ela que transforma uma linha
nova da mesma maneira que o treino foi transformado, e que passa a JSON (para o pacote do
modelo, ADR 0039) e volta, sem pickle.

Uma linha nova chega com as colunas como estavam depois da limpeza (os números como
números, o texto como texto) e passa por:

1. as limpezas que fazem sentido numa linha sozinha (converter um tipo, limitar um
   extremo), com o que aprenderam com os dados;
2. as transformações, com os valores aprendidos no treino;
3. a saída: uma matriz NumPy com as features pela ordem que o modelo conhece.

Sem isto, um modelo guardado não servia para nada: recebia ``Sex = "female"`` e ``Age =
22`` em vez de ``Sex_female = 1`` e ``Age = -0,59`` (a idade já padronizada).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_lab.core.data.boundary import plain_value
from ml_lab.core.data.cleaning.base import AppliedStep, CleaningStep
from ml_lab.core.data.lineage import Lineage
from ml_lab.core.data.schema import ColumnKind, TableSchema
from ml_lab.core.data.transforms.base import transform_class
from ml_lab.core.data.transforms.encoding import as_text, to_zero_one
from ml_lab.core.data.transforms.feature_engineering import DateFeatures
from ml_lab.core.data.transforms.imputation import Imputer
from ml_lab.core.data.transforms.pipeline_builder import TransformPipeline
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.formatting import format_list

MAX_EXAMPLES: Final = 20
"""Quantas linhas do teste ficam à mão, para as linhas de exemplo de um pacote."""

MAX_CATEGORIES: Final = 500
"""Quantas categorias de uma coluna se descrevem, no máximo (as mais frequentes ficam)."""


@dataclass(frozen=True, slots=True)
class InputColumn:
    """Uma coluna que o modelo espera, como chega (antes das transformações).

    Attributes:
        name: O nome da coluna.
        kind: O que ela significa (número, categoria, booleano ou data).
        dtype: O tipo do pandas depois da limpeza.
        minimum: O menor valor visto no treino (números e datas; ``None`` nas outras).
        maximum: O maior valor visto no treino.
        categories: Os valores vistos no treino (categorias e booleanos), por ordem.
        typical: Um valor típico: a mediana, a data do meio ou o valor mais frequente.
        missing_in_train: Quantas linhas de treino não tinham valor.
        can_be_missing: Se o valor pode faltar (a preparação preenche-o).
    """

    name: str
    kind: ColumnKind
    dtype: str
    minimum: float | str | None
    maximum: float | str | None
    categories: tuple[str, ...]
    typical: float | str | bool | None
    missing_in_train: int
    can_be_missing: bool


@dataclass(frozen=True, slots=True, eq=False)
class FittedPreparation:
    """A preparação de um dataset, já treinada, pronta para linhas novas.

    Attributes:
        columns: As colunas que o modelo espera, como chegam (as features antes das
            transformações).
        cleaning: Os passos da limpeza, com o que aprenderam; só os que fazem sentido numa
            linha sozinha se repetem nas linhas novas.
        pipeline: As transformações, já treinadas.
        feature_names: As features que saem, pela ordem que o modelo conhece.
        target: A coluna-alvo (ou a de referência), se houver.
        class_labels: As classes do alvo, na classificação.
        left_out: As colunas da tabela que não entram no modelo.
        examples: Algumas linhas do teste antes das transformações, com o alvo (vazio numa
            preparação que voltou de JSON).
    """

    columns: tuple[InputColumn, ...]
    cleaning: tuple[AppliedStep, ...]
    pipeline: TransformPipeline
    feature_names: tuple[str, ...]
    target: str | None
    class_labels: tuple[str, ...]
    left_out: tuple[str, ...]
    examples: pd.DataFrame

    @property
    def input_names(self) -> tuple[str, ...]:
        """Os nomes das colunas de entrada, pela ordem."""
        return tuple(column.name for column in self.columns)

    def transform(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        """As features de linhas novas, prontas para o modelo.

        Args:
            frame: As linhas, com as colunas de ``input_names`` (outras colunas ignoram-se).

        Raises:
            DataError: Se faltar uma coluna, ou se ficar um valor em falta numa coluna que a
                preparação não preenche.
        """
        absent = [name for name in self.input_names if name not in frame.columns]
        if absent:
            raise DataError(
                f"Faltam colunas para o modelo: {format_list(absent)}. Ele espera "
                f"{format_list(list(self.input_names))}."
            )
        matrix = self._features(frame[list(self.input_names)])
        missing = np.isnan(matrix).any(axis=0)
        if missing.any():
            columns = [self.feature_names[int(i)] for i in np.flatnonzero(missing)]
            raise DataError(
                f"Ficaram valores em falta em {format_list(columns)}: esta preparação não "
                "preenche nulos nessas colunas, por isso cada linha tem de os trazer."
            )
        return matrix

    def input_arrays(self, frame: pd.DataFrame) -> dict[str, NDArray[Any]]:
        """As colunas de linhas novas como as recebe o modelo ONNX: uma matriz ``(N, 1)`` cada.

        Os números (e os booleanos, em 0 ou 1) vão em ``float64``, com ``NaN`` onde faltam; as
        categorias vão em texto, como o one-hot as compara, com o texto vazio onde faltam. As
        datas e o texto livre ficam de fora: não entram no ONNX.
        """
        arrays: dict[str, NDArray[Any]] = {}
        for column in self.columns:
            values = frame[column.name]
            if column.kind is ColumnKind.NUMERIC:
                numbers = values.astype("float64").to_numpy(dtype=np.float64, na_value=np.nan)
                arrays[column.name] = numbers.reshape(-1, 1)
            elif column.kind is ColumnKind.BOOLEAN:
                arrays[column.name] = to_zero_one(values).to_numpy(dtype=np.float64).reshape(-1, 1)
            elif column.kind is ColumnKind.CATEGORICAL:
                text = as_text(values).fillna("").to_numpy(dtype=object)
                arrays[column.name] = text.reshape(-1, 1)
        return arrays

    def check_rows(self) -> tuple[dict[str, NDArray[Any]], NDArray[np.float64]]:
        """As linhas de exemplo que a preparação transforma, para confirmar uma conversão.

        Saem como as recebe o ONNX (``input_arrays``) e como features (o que o modelo recebe
        no laboratório). Uma linha com um valor em falta que a preparação não preenche fica de
        fora: nem o laboratório a sabe prever.
        """
        if self.examples.empty:
            return {}, np.zeros((0, len(self.feature_names)))
        rows = self.examples[list(self.input_names)]
        matrix = self._features(rows)
        complete = ~np.isnan(matrix).any(axis=1)
        return self.input_arrays(rows[complete]), matrix[complete]

    def _features(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        """As features das linhas, pela ordem do modelo (com ``NaN`` onde ficar a faltar)."""
        for step in self.cleaning:
            rows = step.replay(rows)
        transformed = self.pipeline.transform(rows)[list(self.feature_names)]
        matrix: NDArray[np.float64] = transformed.to_numpy(dtype=np.float64, na_value=np.nan)
        return matrix

    def input_document(self) -> dict[str, Any]:
        """O ``input.json`` de um pacote: as colunas que o modelo espera e o alvo."""
        target = (
            None
            if self.target is None
            else {"name": self.target, "classes": list(self.class_labels)}
        )
        return {
            "target": target,
            "columns": [_column_document(column) for column in self.columns],
            "left_out": list(self.left_out),
        }

    def preparation_document(self) -> dict[str, Any]:
        """O ``preparation.json`` de um pacote: a receita e o que ela aprendeu, por ordem."""
        return {
            "cleaning": [
                {
                    "operation": applied.step.operation,
                    "params": _plain(applied.step.params),
                    "learned": _plain(applied.learned),
                    "on_new_rows": applied.replayable,
                }
                for applied in self.cleaning
            ],
            "transforms": [
                {
                    "operation": step.name,
                    "title": step.title,
                    "params": _plain(step.params()),
                    "learned": _plain(step.learned()),
                    "description": step.describe(),
                }
                for step in self.pipeline.steps
            ],
            "features": list(self.feature_names),
        }

    @classmethod
    def from_documents(
        cls, inputs: Mapping[str, Any], preparation: Mapping[str, Any]
    ) -> "FittedPreparation":
        """Volta a criar a preparação a partir do ``input.json`` e do ``preparation.json``.

        Raises:
            InvalidParameterError: Se um dos documentos estiver incompleto ou tiver uma
                transformação desconhecida.
        """
        try:
            target = inputs.get("target")
            columns = tuple(_column_from(document) for document in inputs["columns"])
            cleaning = tuple(
                AppliedStep(
                    CleaningStep(step["operation"], dict(step.get("params", {}))),
                    dict(step.get("learned", {})),
                )
                for step in preparation.get("cleaning", [])
            )
            steps = [
                transform_class(step["operation"]).restore(step["params"], step["learned"])
                for step in preparation["transforms"]
            ]
            return cls(
                columns=columns,
                cleaning=cleaning,
                pipeline=TransformPipeline(steps),
                feature_names=tuple(str(name) for name in preparation["features"]),
                target=None if target is None else str(target["name"]),
                class_labels=() if target is None else tuple(map(str, target["classes"])),
                left_out=tuple(str(name) for name in inputs.get("left_out", [])),
                examples=pd.DataFrame(),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidParameterError(
                f"A preparação guardada está incompleta e não volta a abrir: falta {error}."
            ) from error

    def examples_csv(self, predicted: Sequence[object], n_rows: int) -> str:
        """As primeiras ``n_rows`` linhas do teste em CSV, com o valor real e o previsto.

        Args:
            predicted: O que o modelo previu para essas linhas, pela mesma ordem.
            n_rows: Quantas linhas (no máximo, as guardadas).
        """
        rows = self.examples.head(n_rows).copy()
        rows["previsto"] = pd.Series(list(predicted)[: len(rows)], index=rows.index, dtype=object)
        text: str = rows.to_csv(index=False, lineterminator="\n")
        return text


def fit_preparation(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    schema: TableSchema,
    features: Sequence[str],
    pipeline: TransformPipeline,
    feature_names: Sequence[str],
    cleaning: Sequence[AppliedStep],
    target: str | None,
    class_labels: Sequence[str],
    table_columns: Sequence[str],
) -> FittedPreparation:
    """Junta o que a preparação aprendeu (chamada no fim da preparação, com o treino em bruto).

    Args:
        train: As linhas de treino depois da limpeza, antes das transformações.
        test: As linhas de teste, da mesma maneira (para as linhas de exemplo).
        schema: O esquema das colunas.
        features: As colunas que entram no modelo.
        pipeline: As transformações, já treinadas.
        feature_names: As features que saem das transformações.
        cleaning: Os passos da limpeza aplicados, com o que aprenderam.
        target: A coluna-alvo (ou a de referência).
        class_labels: As classes do alvo.
        table_columns: Todas as colunas da tabela carregada.
    """
    imputed = {
        str(column)
        for step in pipeline.steps
        if isinstance(step, Imputer)
        for column in step.columns
    }
    kinds = {column.name: column.kind for column in schema.columns}
    columns = tuple(_profile(train[name], name, kinds[name], imputed) for name in features)
    used = {*features, *([target] if target is not None else [])}
    kept = [*features, *([target] if target is not None and target in test.columns else [])]
    return FittedPreparation(
        columns=columns,
        cleaning=tuple(cleaning),
        pipeline=pipeline,
        feature_names=tuple(feature_names),
        target=target,
        class_labels=tuple(class_labels),
        left_out=tuple(str(name) for name in table_columns if str(name) not in used),
        examples=test[kept].head(MAX_EXAMPLES).reset_index(drop=True),
    )


def lineage_document(lineage: Lineage) -> list[dict[str, Any]]:
    """A linhagem em valores de JSON, um dicionário por passo."""
    return [_plain(record) for record in lineage.to_records()]


def _profile(values: pd.Series, name: str, kind: ColumnKind, imputed: set[str]) -> InputColumn:
    """O perfil de uma coluna de entrada, a partir das linhas de treino."""
    present = values.dropna()
    minimum = maximum = typical = None
    categories: tuple[str, ...] = ()
    if kind in (ColumnKind.NUMERIC, ColumnKind.DATETIME) and not present.empty:
        minimum, maximum = plain_value(present.min()), plain_value(present.max())
        typical = plain_value(present.median())
    elif not present.empty:
        counts = present.astype("str").value_counts()
        categories = tuple(sorted(str(value) for value in counts.index[:MAX_CATEGORIES]))
        typical = plain_value(present.mode().iloc[0])
    if kind is ColumnKind.DATETIME:
        # Uma data pode faltar se todas as suas partes forem preenchidas.
        can_be_missing = set(DateFeatures.output_names(name)) <= imputed
    else:
        can_be_missing = name in imputed
    return InputColumn(
        name=name,
        kind=kind,
        dtype=str(values.dtype),
        minimum=minimum,
        maximum=maximum,
        categories=categories,
        typical=typical,
        missing_in_train=int(values.isna().sum()),
        can_be_missing=can_be_missing,
    )


def _column_document(column: InputColumn) -> dict[str, Any]:
    document: dict[str, Any] = {
        "name": column.name,
        "kind": column.kind.value,
        "dtype": column.dtype,
        "missing_in_train": column.missing_in_train,
        "can_be_missing": column.can_be_missing,
        "typical": column.typical,
    }
    if column.kind in (ColumnKind.NUMERIC, ColumnKind.DATETIME):
        document |= {"min": column.minimum, "max": column.maximum}
    else:
        document["categories"] = list(column.categories)
    return document


def _column_from(document: Mapping[str, Any]) -> InputColumn:
    kind = ColumnKind(document["kind"])
    return InputColumn(
        name=str(document["name"]),
        kind=kind,
        dtype=str(document.get("dtype", "object")),
        minimum=document.get("min"),
        maximum=document.get("max"),
        categories=tuple(str(value) for value in document.get("categories", [])),
        typical=document.get("typical"),
        missing_in_train=int(document.get("missing_in_train", 0)),
        can_be_missing=bool(document.get("can_be_missing", False)),
    )


def _plain(value: Any) -> Any:
    """``value`` em valores simples de JSON: dicionários, listas, texto, números e ``None``."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return plain_value(value)
