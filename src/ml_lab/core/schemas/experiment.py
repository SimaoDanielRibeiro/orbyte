"""Configuração de uma experiência: dados, preparação, modelo e seed (Pydantic).

É o que entra no serviço de experiências e o que fica guardado com cada run: com a
mesma configuração obtém-se o mesmo resultado. Vai e volta de JSON sem perder nada,
por isso serve também para a API (fase 7) e para o registo de experiências (fase 5).

A origem dos dados e a preparação são as dataclasses da camada de dados
(``SourceSpec`` e ``PreparationConfig``): o Pydantic valida-as e serializa-as tal
como estão, sem as duplicar (ADR 0011).

Exemplo:
    >>> config = ExperimentConfig.model_validate(
    ...     {
    ...         "dataset": {"kind": "sample", "params": {"name": "iris"}},
    ...         "preparation": {"target": "species"},
    ...         "model": {"name": "logistic_regression", "hyperparameters": {"C": 0.5}},
    ...         "seed": 42,
    ...     }
    ... )
    >>> config.preparation.split.test_size
    0.2
    >>> ExperimentConfig.model_validate_json(config.model_dump_json()) == config
    True
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, with_config

from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.projection.projection_2d import ProjectionConfig
from ml_lab.core.data.splitting.base import SplitConfig
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig
from ml_lab.core.foundation.random import MAX_SEED

# As dataclasses da camada de dados não conhecem o Pydantic (ADR 0011): é aqui, na fronteira,
# que se diz como as validar. Sem uma configuração própria, cada uma herdava a do modelo onde
# aparece: recusava campos a mais dentro de um ExperimentConfig e aceitava-os dentro de uma
# CleaningSuggestion devolvida pelo serviço. O OpenAPI ficava então com duas versões de cada
# uma ("CleaningStep-Input" e "CleaningStep-Output"), que dariam dois tipos no TypeScript do
# frontend (ADR 0026).
for _dataclass in (
    SourceSpec,
    PreparationConfig,
    CleaningStep,
    SplitConfig,
    TransformConfig,
    ProjectionConfig,
):
    with_config(ConfigDict(extra="forbid"))(_dataclass)


class ModelSpec(BaseModel):
    """O modelo escolhido e os seus hiperparâmetros.

    Os hiperparâmetros são validados pelo próprio modelo (cada um tem os seus), por
    isso aqui são só um dicionário.

    Attributes:
        name: Nome do modelo no registo, por exemplo ``"logistic_regression"``.
        hyperparameters: Valores escolhidos; os que faltam usam o valor por omissão.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Tudo o que define uma experiência.

    Attributes:
        dataset: De onde vêm os dados.
        preparation: Alvo, tarefa, limpeza, divisão e transformações.
        model: O modelo e os hiperparâmetros.
        seed: Seed raiz; ``None`` gera uma nova, que fica registada no resultado.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: SourceSpec
    preparation: PreparationConfig = Field(default_factory=PreparationConfig)
    model: ModelSpec
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
