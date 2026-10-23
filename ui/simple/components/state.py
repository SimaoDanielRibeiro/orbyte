"""Estado da sessão: os serviços e a experiência atual.

O Streamlit volta a correr o script a cada interação, por isso tudo o que tem de
sobreviver entre interações vive em ``st.session_state``: o ``Lab`` com os
serviços (um por sessão), o dataset escolhido, a última preparação feita, o modelo
escolhido e o resultado do último treino.
"""

from dataclasses import dataclass
from typing import Final

import streamlit as st

from ml_lab.core.services import (
    DatasetSummary,
    ExperimentResult,
    Lab,
    MLLabError,
    ModelCard,
    PreparationConfig,
    PreparationSummary,
)

_LAB: Final = "lab"
_DATASET: Final = "dataset_id"
_PREPARATION: Final = "preparation"
_MODEL: Final = "model"
_RESULT: Final = "result"


@dataclass(frozen=True, slots=True)
class Preparation:
    """A última preparação feita no ecrã Preparar.

    Attributes:
        dataset_id: O dataset preparado.
        config: A configuração usada.
        seed: A seed da experiência.
        summary: O resultado.
    """

    dataset_id: str
    config: PreparationConfig
    seed: int
    summary: PreparationSummary


def get_lab() -> Lab:
    """Os serviços desta sessão (criados na primeira vez)."""
    if _LAB not in st.session_state:
        st.session_state[_LAB] = Lab.create()
    lab: Lab = st.session_state[_LAB]
    return lab


def current_dataset() -> DatasetSummary | None:
    """O dataset escolhido, se houver."""
    dataset_id: str | None = st.session_state.get(_DATASET)
    if dataset_id is None:
        return None
    try:
        return get_lab().datasets.summary(dataset_id)
    except MLLabError:  # saiu da memória do serviço: volta a ser preciso carregá-lo
        st.session_state.pop(_DATASET, None)
        return None


def select_dataset(summary: DatasetSummary) -> None:
    """Escolhe o dataset atual; uma preparação de outro dataset deixa de valer."""
    if st.session_state.get(_DATASET) != summary.id:
        st.session_state.pop(_PREPARATION, None)
    st.session_state[_DATASET] = summary.id


def current_preparation() -> Preparation | None:
    """A última preparação, se for do dataset atual."""
    preparation: Preparation | None = st.session_state.get(_PREPARATION)
    if preparation is None or preparation.dataset_id != st.session_state.get(_DATASET):
        return None
    return preparation


def save_preparation(preparation: Preparation) -> None:
    """Guarda a preparação feita."""
    st.session_state[_PREPARATION] = preparation


def clear_preparation() -> None:
    """Esquece a preparação (por exemplo, quando a configuração deixa de ser válida)."""
    st.session_state.pop(_PREPARATION, None)


def chosen_model() -> ModelCard | None:
    """O modelo escolhido no ecrã Modelo, se houver."""
    card: ModelCard | None = st.session_state.get(_MODEL)
    return card


def choose_model(card: ModelCard) -> None:
    """Guarda o modelo escolhido."""
    st.session_state[_MODEL] = card


def last_result() -> ExperimentResult | None:
    """O resultado do último treino, se houver."""
    result: ExperimentResult | None = st.session_state.get(_RESULT)
    return result


def save_result(result: ExperimentResult) -> None:
    """Guarda o resultado de um treino."""
    st.session_state[_RESULT] = result
