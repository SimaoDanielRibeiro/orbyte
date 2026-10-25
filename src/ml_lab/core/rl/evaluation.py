"""As métricas de um agente: no treino, e com a política aprendida (sem explorar).

Um agente não tem dados de teste, mas a pergunta é a mesma: o que aprendeu serve? Por isso
o relatório tem as duas partes de sempre, com o mesmo sentido:

- **treino**: os últimos ``TRAINING_WINDOW`` episódios do treino, ainda com exploração. Mostram
  como o agente se portava enquanto aprendia;
- **teste**: ``EVALUATION_EPISODES`` episódios novos com a política aprendida, sem explorar e
  sem aprender. É o que o agente sabe fazer.

A diferença entre as duas partes é o preço de explorar. No penhasco, o agente cai nas
armadilhas durante o treino (o ε escolhe ao acaso, mesmo colado a elas), mas a política
aprendida atravessa sem cair.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from ml_lab.core.evaluation.report import EvaluationReport, MetricResult
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.rl.agents.base import Agent
from ml_lab.core.rl.environments.base import Environment
from ml_lab.core.training.recorder import StepRecord
from ml_lab.formatting import format_percent

EVALUATION_EPISODES: Final = 20
"""Episódios com a política aprendida (o teste)."""

TRAINING_WINDOW: Final = 100
"""Os últimos episódios do treino que entram no relatório do treino."""

_SUCCESS: Final = (
    "A proporção de episódios que acabaram no objetivo.",
    "No teste, sem explorar: 100% quer dizer que a política aprendida chega sempre lá. Num "
    "ambiente onde se escorrega, nenhuma política chega sempre.",
)
_REWARD: Final = (
    "A soma das recompensas de um episódio, em média: +1 no objetivo, −1 numa armadilha e "
    "−0,01 por passo.",
    "Quanto maior, melhor: perto de 1 é chegar ao objetivo pelo caminho curto.",
)
_LENGTH: Final = (
    "Quantos passos dura um episódio, em média.",
    "Lê-se com a taxa de sucesso: poucos passos e sucesso é o caminho curto; poucos passos sem "
    "sucesso é cair numa armadilha; muitos passos é andar às voltas.",
)


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Como acabou um episódio.

    Attributes:
        reward: A soma das recompensas.
        length: Quantos passos durou.
        reached: Se acabou no objetivo.
    """

    reward: float
    length: int
    reached: bool


def play(
    agent: Agent[Any], environment: Environment[Any], episodes: int, seed: int
) -> list[EpisodeOutcome]:
    """Joga ``episodes`` episódios com a política aprendida, sem explorar e sem aprender."""
    rng = np.random.default_rng(seed)
    outcomes = []
    for _ in range(episodes):
        state = environment.reset(rng)
        total, reached, length = 0.0, False, environment.max_steps
        for step in range(1, environment.max_steps + 1):
            transition = environment.step(state, agent.act(state), rng)
            total += transition.reward
            state = transition.state
            if transition.done:
                reached, length = transition.reached_goal, step
                break
        outcomes.append(EpisodeOutcome(total, length, reached))
    return outcomes


def training_outcomes(history: Sequence[StepRecord]) -> list[EpisodeOutcome]:
    """Os últimos ``TRAINING_WINDOW`` episódios do treino, como o contexto os registou."""
    return [
        EpisodeOutcome(
            record.metrics["reward"],
            int(record.metrics["length"]),
            record.metrics["success"] > 0,
        )
        for record in history[-TRAINING_WINDOW:]
    ]


def episodes_report(outcomes: Sequence[EpisodeOutcome], partition: Partition) -> EvaluationReport:
    """O relatório de um conjunto de episódios: o sucesso, a recompensa e a duração."""
    return EvaluationReport(
        partition=partition,
        task_type=TaskType.CONTROL,
        n_samples=len(outcomes),
        metrics=(
            MetricResult(
                "success_rate",
                "Taxa de sucesso",
                _mean([float(outcome.reached) for outcome in outcomes]),
                True,
                True,
                *_SUCCESS,
            ),
            MetricResult(
                "mean_reward",
                "Recompensa média",
                _mean([outcome.reward for outcome in outcomes]),
                True,
                False,
                *_REWARD,
            ),
            MetricResult(
                "mean_length",
                "Passos por episódio",
                _mean([float(outcome.length) for outcome in outcomes]),
                None,
                False,
                *_LENGTH,
            ),
        ),
    )


def reading(train: EvaluationReport, test: EvaluationReport, slip: float) -> tuple[Notice, ...]:
    """O que a diferença entre o treino e o teste diz."""
    trained, learned = _success(train), _success(test)
    notices = []
    if learned < 1 and slip == 0:
        notices.append(
            Notice.warning(
                "not_solved",
                f"A política aprendida só chega ao objetivo em {format_percent(learned)} dos "
                "episódios, num ambiente sem acaso: ainda não aprendeu o caminho. Experimenta "
                "mais episódios.",
            )
        )
    if learned - trained >= 0.1:
        notices.append(
            Notice.info(
                "exploration_cost",
                f"No treino, o agente chegou ao objetivo em {format_percent(trained)} dos "
                f"episódios; com a política aprendida, chega em {format_percent(learned)}. A "
                "diferença é o preço de explorar: o ε continua a escolher ao acaso, até ao fim.",
            )
        )
    return tuple(notices)


def _success(report: EvaluationReport) -> float:
    value = report.metric("success_rate").value
    return 0.0 if value is None else value


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None
