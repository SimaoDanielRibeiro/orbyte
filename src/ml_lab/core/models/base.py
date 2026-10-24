"""Contrato LabModel e ficha técnica do modelo.

Um modelo do ML Lab é uma classe que:

- descreve-se a si própria numa ficha técnica (``ModelInfo``): nome, tipos de tarefa,
  família, implementação, se é iterativo, se dá probabilidades, hiperparâmetros e
  ficha didática. É a ficha que monta o catálogo e gera os formulários;
- recebe hiperparâmetros validados (Pydantic) e uma seed;
- treina com ``fit(X, y, contexto)`` e prevê com ``predict(X)`` (e, se a ficha o
  disser, ``predict_proba(X)``).

Os dados chegam sempre como matrizes NumPy ``float64``; na classificação, o alvo é o
número de cada classe (0, 1, 2…). O modelo não conhece o pandas nem a interface.

Para acrescentar um modelo, ver "Adicionar um algoritmo" em ``docs/architecture.md``:
um ficheiro na pasta do paradigma, decorado com ``@MODELS.register("nome")``.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Self, cast

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.errors import CapabilityError, ModelError, NotFittedError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.random import validate_seed
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import NullTrainingContext, TrainingContext

type Features = NDArray[np.float64]
type Target = NDArray[np.int64] | NDArray[np.float64]


class ModelFamily(StrEnum):
    """Família do modelo, para agrupar o catálogo. Cresce com os modelos da fase 4."""

    LINEAR = "linear"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _FAMILY_LABELS[self]


class Implementation(StrEnum):
    """Quem implementa o algoritmo."""

    DIDACTIC = "didactic"
    SKLEARN = "sklearn"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _IMPLEMENTATION_LABELS[self]


_FAMILY_LABELS: Final[Mapping[ModelFamily, str]] = MappingProxyType(
    {ModelFamily.LINEAR: "Modelos lineares"}
)
_IMPLEMENTATION_LABELS: Final[Mapping[Implementation, str]] = MappingProxyType(
    {Implementation.DIDACTIC: "Didática (NumPy)", Implementation.SKLEARN: "scikit-learn"}
)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Ficha técnica de um modelo.

    Attributes:
        name: Identificador, igual ao nome no registo ``MODELS``.
        display_name: Nome a mostrar.
        summary: Uma frase sobre o que o modelo faz, para o cartão do catálogo.
        task_types: Tipos de tarefa que resolve.
        family: Família, para agrupar o catálogo.
        implementation: Didática ou de uma biblioteca.
        iterative: Se treina por passos (e por isso anuncia passos e snapshots).
        supports_proba: Se dá probabilidades (permite curvas ROC e a log-loss).
        hyperparameters: A classe dos hiperparâmetros.
        doc: A ficha didática, em ``docs/algorithms/``.
        step_limit: Nos modelos iterativos, o hiperparâmetro que limita o número de passos
            (por exemplo ``max_epochs``): é o que permite mostrar uma barra de progresso.
    """

    name: str
    display_name: str
    summary: str
    task_types: frozenset[TaskType]
    family: ModelFamily
    implementation: Implementation
    iterative: bool
    supports_proba: bool
    hyperparameters: type[Hyperparameters]
    doc: str
    step_limit: str | None = None


class LabModel[P: Hyperparameters](ABC):
    """Contrato de um modelo do ML Lab.

    As subclasses definem ``info`` e implementam ``_fit`` e ``_predict`` (e
    ``_predict_proba``, se ``info.supports_proba``). As verificações comuns (formas
    dos arrays, modelo já treinado, capacidades) ficam aqui, uma só vez.

    Args:
        hyperparameters: Os hiperparâmetros, da classe ``info.hyperparameters``; por
            omissão, os valores por omissão.
        seed: Seed do modelo (normalmente ``seeds.seed_for("model")``).
    """

    info: ClassVar[ModelInfo]

    def __init__(self, hyperparameters: P | None = None, *, seed: int = 0) -> None:
        expected = self.info.hyperparameters
        chosen = hyperparameters if hyperparameters is not None else expected()
        if not isinstance(chosen, expected):
            raise ModelError(
                f"'{self.info.name}' recebe hiperparâmetros do tipo {expected.__name__}, "
                f"não {type(chosen).__name__}."
            )
        self.hyperparameters = cast("P", chosen)
        self.seed = validate_seed(seed)
        self._n_features: int | None = None
        self._notices: list[Notice] = []

    @property
    def is_fitted(self) -> bool:
        """Se o modelo já treinou."""
        return self._n_features is not None

    @property
    def notices(self) -> tuple[Notice, ...]:
        """Avisos do treino (por exemplo, o otimizador não convergiu)."""
        return tuple(self._notices)

    def fit(
        self, X: Features, y: Target | None = None, context: TrainingContext | None = None
    ) -> Self:
        """Treina o modelo.

        Args:
            X: Features de treino, ``(n_linhas, n_features)``.
            y: Alvo de treino, ``(n_linhas,)``; ``None`` nas tarefas sem alvo.
            context: Para onde anunciar os passos do treino.

        Raises:
            ModelError: Se as formas dos arrays não forem as esperadas.
        """
        if X.ndim != 2 or X.shape[0] == 0:
            raise ModelError(
                f"X tem de ser uma matriz (linhas × features) com linhas; tem forma {X.shape}."
            )
        if y is not None and y.shape != (X.shape[0],):
            raise ModelError(
                f"y tem de ter um valor por linha de X: {y.shape} em vez de ({X.shape[0]},)."
            )
        self._notices = []
        self._fit(X, y, context or NullTrainingContext())
        self._n_features = X.shape[1]
        return self

    def predict(self, X: Features) -> NDArray[Any]:
        """Previsões para ``X``: classes (0, 1, 2…) ou valores, conforme a tarefa."""
        self._check_ready(X)
        return np.asarray(self._predict(X))

    def predict_proba(self, X: Features) -> NDArray[np.float64]:
        """Probabilidade de cada classe vista no treino (``classes()``), por linha.

        Raises:
            CapabilityError: Se o modelo não der probabilidades.
        """
        if not self.info.supports_proba:
            raise CapabilityError(f"'{self.info.display_name}' não dá probabilidades.")
        self._check_ready(X)
        return np.asarray(self._predict_proba(X), dtype=np.float64)

    def classes(self) -> NDArray[np.int64] | None:
        """Nos classificadores, as classes vistas no treino; as colunas de ``predict_proba``."""
        return None

    def state(self) -> dict[str, Any]:
        """O que o modelo aprendeu, em valores simples (coeficientes, centróides...)."""
        return {}

    def snapshot_state(self, probe: Features | None = None) -> dict[str, Any]:
        """O estado agora, mais as previsões nos pontos da sonda (se houver).

        Pode ser chamado a meio do treino, porque usa os métodos internos: é assim que um
        modelo iterativo fotografa o seu estado em cada snapshot.
        """
        state = self.state()
        if probe is not None:
            values = self._probe_values(probe)
            if values is not None:
                state["probe"] = values
        return state

    def _note(self, notice: Notice) -> None:
        """Guarda um aviso do treino."""
        self._notices.append(notice)

    def _check_ready(self, X: Features) -> None:
        if self._n_features is None:
            raise NotFittedError(
                f"'{self.info.display_name}' ainda não treinou: chama fit antes de prever."
            )
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ModelError(
                f"O modelo treinou com {self._n_features} features e recebeu {X.shape} para prever."
            )

    @abstractmethod
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        """Treina (as formas já foram confirmadas)."""

    @abstractmethod
    def _predict(self, X: Features) -> NDArray[Any]:
        """Prevê (o modelo já treinou e ``X`` tem o número certo de features)."""

    def _predict_proba(self, X: Features) -> NDArray[np.float64]:
        raise CapabilityError(f"'{self.info.display_name}' não implementa predict_proba.")

    def _probe_values(self, probe: Features) -> NDArray[np.float64] | None:
        """As previsões nos pontos da sonda: probabilidades, se as houver; senão, previsões."""
        if self.info.supports_proba:
            return np.asarray(self._predict_proba(probe), dtype=np.float64)
        return np.asarray(self._predict(probe), dtype=np.float64)


MODELS: Registry[type[LabModel[Any]]] = Registry(
    "models", task_types=lambda model: model.info.task_types, discover="ml_lab.core.models"
)
"""Todos os modelos, pelo nome. A descoberta importa cada ficheiro de ``core/models/``."""


def capture_warnings[T](action: Callable[[], T]) -> tuple[T, list[warnings.WarningMessage]]:
    """Corre ``action`` e devolve o resultado e os avisos que as bibliotecas emitiram.

    Os avisos não se perdem nem rebentam: o modelo transforma-os em avisos didáticos.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = action()
    return result, list(caught)
