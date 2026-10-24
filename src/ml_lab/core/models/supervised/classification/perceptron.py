"""Perceptrão: implementação didática em NumPy do classificador linear mais antigo (1958).

Para cada exemplo, calcula ``w · x + b``; se o sinal estiver errado, corrige os pesos
na direção do exemplo: ``w ← w + taxa · y · x`` (com ``y`` = +1 ou -1). Se acertar, não
mexe em nada. Só aprende com os erros, e só pára de errar se as classes se separarem com
uma reta (é o teorema de convergência do perceptrão).

Com mais de duas classes, treina um perceptrão por classe ("esta classe contra as
outras") e prevê a classe com a pontuação mais alta, como o scikit-learn. Ficha
didática: ``docs/algorithms/perceptron.md``.
"""

from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    LabModel,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext
from ml_lab.formatting import format_count


class PerceptronParams(Hyperparameters):
    """Hiperparâmetros do perceptrão."""

    learning_rate: float = Field(
        default=1.0,
        ge=0.001,
        le=10,
        title="Taxa de aprendizagem",
        description=(
            "Tamanho de cada correção. Como os pesos começam a zero, só muda a escala dos "
            "pesos, não a fronteira; importa quando há ordenada."
        ),
        json_schema_extra={"x-scale": "log"},
    )
    max_epochs: int = Field(
        default=100,
        ge=1,
        le=1_000,
        title="Máximo de épocas",
        description="Quantas passagens pelos dados de treino, no máximo.",
    )
    shuffle: bool = Field(
        default=True,
        title="Baralhar a cada época",
        description=(
            "Percorrer os exemplos por uma ordem diferente em cada época. A fronteira final "
            "depende da ordem; baralhar evita ciclos em que os mesmos erros se repetem."
        ),
    )
    fit_intercept: bool = Field(
        default=True,
        title="Aprender a ordenada (b)",
        description="Sem ordenada, a fronteira é obrigada a passar pela origem.",
    )


@MODELS.register("perceptron")
class Perceptron(LabModel[PerceptronParams]):
    """Perceptrão de Rosenblatt, com uma época por passo do treino."""

    info = ModelInfo(
        name="perceptron",
        display_name="Perceptrão",
        summary=(
            "O primeiro classificador que aprende: uma reta que só se corrige quando erra. Vês "
            "a fronteira a saltar a cada época até separar as classes (se der)."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.DIDACTIC,
        iterative=True,
        supports_proba=False,
        hyperparameters=PerceptronParams,
        doc="perceptron.md",
        step_limit="max_epochs",
    )

    def __init__(self, hyperparameters: PerceptronParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._classes = np.zeros(0, dtype=np.int64)
        self._weights = np.zeros((0, 0))
        self._bias = np.zeros(0)
        self._epochs = 0
        self._separated = False

    @override
    def classes(self) -> NDArray[np.int64] | None:
        return self._classes if self._classes.size else None

    @override
    def state(self) -> dict[str, Any]:
        return {
            "coefficients": self._weights.tolist(),
            "intercept": self._bias.tolist(),
            "classes": self._classes.tolist(),
            "iterations": self._epochs,
            "converged": self._separated,
        }

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if y is None:
            raise ModelError("O perceptrão precisa do alvo para aprender.")
        params = self.hyperparameters
        self._classes = np.unique(y).astype(np.int64)
        if len(self._classes) < 2:
            raise ModelError("O perceptrão precisa de pelo menos duas classes no treino.")
        # +1 para "é desta classe", -1 para "não é"; com duas classes, um só perceptrão.
        codes = np.searchsorted(self._classes, y)
        n_outputs = 1 if len(self._classes) == 2 else len(self._classes)
        if n_outputs == 1:
            signs = np.where(codes == 1, 1.0, -1.0)[:, None]
        else:
            signs = np.where(codes[:, None] == np.arange(n_outputs), 1.0, -1.0)
        n_samples, n_features = X.shape
        self._weights = np.zeros((n_outputs, n_features))
        self._bias = np.zeros(n_outputs)
        self._epochs = 0
        self._separated = False
        rng = np.random.default_rng(self.seed)
        rate = params.learning_rate

        for epoch in range(1, params.max_epochs + 1):
            order = rng.permutation(n_samples) if params.shuffle else np.arange(n_samples)
            mistakes = 0
            for index in order:
                x, sign = X[index], signs[index]
                wrong = sign * (self._weights @ x + self._bias) <= 0
                if wrong.any():
                    mistakes += 1
                    self._weights[wrong] += rate * sign[wrong, None] * x
                    if params.fit_intercept:
                        self._bias[wrong] += rate * sign[wrong]
            self._epochs = epoch
            accuracy = float(np.mean(self._predict(X) == y))
            context.report_step(epoch, loss=mistakes / n_samples, metrics={"accuracy": accuracy})
            if context.wants_snapshot(epoch):
                context.report_snapshot(epoch, self.snapshot_state(context.probe))
            if mistakes == 0:
                self._separated = True
                return
            if context.should_stop():
                return
        self._note(
            Notice.info(
                "not_separated",
                f"Ao fim de {format_count(params.max_epochs)} épocas, o perceptrão ainda errava "
                "no treino: as classes não se separam com uma reta (ou precisava de mais "
                "épocas). Ficou com os pesos da última época.",
            )
        )

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        scores = X @ self._weights.T + self._bias
        if scores.shape[1] == 1:
            return self._classes[(scores[:, 0] > 0).astype(np.int64)]
        return self._classes[np.argmax(scores, axis=1)]
