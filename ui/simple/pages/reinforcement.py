"""Ecrã 11 · Reinforcement learning: um agente aprende a atravessar uma grelha."""

import streamlit as st

from charts.reinforcement import live_reward_chart, policy_chart, q_table_chart, reward_chart
from components.forms import schema_form
from components.layout import render_header, render_why
from components.notices import render_error, render_notices
from components.state import current_rl_job, get_lab, last_rl_result, save_rl_job
from ml_lab.core.services import (
    AgentSpec,
    EnvironmentSpec,
    JobStatus,
    MLLabError,
    Partition,
    PolicyFrame,
    ReinforcementConfig,
    ReinforcementResult,
    TrainingJob,
    explain,
)
from ml_lab.formatting import format_count, format_number

# De quanto em quanto tempo (em segundos) o ecrã volta a ler o progresso do treino.
_REFRESH_SECONDS = 0.5


def main() -> None:
    """Desenha o ecrã."""
    render_header("reinforcement")
    job = current_rl_job()
    if job is not None and not job.done:
        _live(job)
        return
    if job is not None and job.error is not None:
        render_error(job.error)
    result = last_rl_result()
    if result is None:
        _setup()
    else:
        _result(result)
        with st.expander("Treinar outro agente", icon=":material/smart_toy:"):
            _setup()
    render_why(
        "reinforcement_learning",
        "q_values",
        "temporal_difference",
        "exploration",
        "discount",
        "deep_q_network",
        "rl_evaluation",
    )


def _setup() -> None:
    """O ambiente, o agente e os hiperparâmetros; e o botão que começa o treino."""
    lab = get_lab()
    environments = {card.name: card for card in lab.reinforcement.environments()}
    agents = {card.name: card for card in lab.reinforcement.agents()}
    environment_column, agent_column = st.columns(2)
    with environment_column:
        st.markdown("**Ambiente**")
        environment = st.selectbox(
            "Ambiente",
            list(environments),
            format_func=lambda name: environments[name].display_name,
            key="rl.environment",
            label_visibility="collapsed",
        )
        st.caption(environments[environment].summary)
        parameters = schema_form(
            environments[environment].parameters_schema, key=f"rl.environment.{environment}"
        )
    with agent_column:
        st.markdown("**Agente**")
        agent = st.radio(
            "Agente",
            list(agents),
            format_func=lambda name: agents[name].display_name,
            key="rl.agent",
            horizontal=True,
            label_visibility="collapsed",
        )
        card = agents[agent]
        st.caption(f"{card.implementation.label} · {card.summary}")
        with st.expander("Ficha didática", icon=":material/menu_book:"):
            st.markdown(card.doc or "A ficha didática deste agente ainda não foi escrita.")
    st.markdown("**Hiperparâmetros**")
    hyperparameters = schema_form(card.hyperparameters_schema, key=f"rl.agent.{agent}")
    seed = st.number_input(
        "Seed",
        min_value=0,
        value=42,
        step=1,
        key="rl.seed",
        help="A mesma seed dá o mesmo treino: a mesma exploração e o mesmo acaso do ambiente.",
    )
    if st.button("Treinar o agente", type="primary", icon=":material/play_arrow:", key="rl.train"):
        config = ReinforcementConfig(
            environment=EnvironmentSpec(name=environment, parameters=parameters),
            agent=AgentSpec(name=agent, hyperparameters=hyperparameters),
            seed=int(seed),
        )
        try:
            job = lab.reinforcement.start(config)
        except MLLabError as error:  # configuração inválida: o treino nem chega a começar
            render_error(error)
            return
        save_rl_job(job)
        st.rerun()


@st.fragment(run_every=_REFRESH_SECONDS)
def _live(job: TrainingJob[ReinforcementResult]) -> None:
    """O treino a decorrer: só esta parte do ecrã se atualiza, a cada meio segundo."""
    if job.done:
        st.rerun()  # o ecrã inteiro volta a correr e mostra o resultado
    update = get_lab().reinforcement.updates(job.run_id)
    state = update.state
    pause, stop, _ = st.columns([1, 1, 4])
    if state.status is JobStatus.PAUSED:
        pause.button("Retomar", icon=":material/play_arrow:", on_click=job.resume, key="rl.resume")
    else:
        pause.button("Pausar", icon=":material/pause:", on_click=job.pause, key="rl.pause")
    stop.button(
        "Parar",
        icon=":material/stop:",
        on_click=job.cancel,
        key="rl.stop",
        help="O treino pára no fim do episódio atual; o agente fica com o que aprendeu.",
    )
    episodes, planned = state.progress.steps, state.planned_steps
    text = f"{state.status.label} · episódio {format_count(episodes)}"
    if planned:
        st.progress(min(1.0, episodes / planned), text=f"{text} de {format_count(planned)}")
    else:
        st.markdown(text)
    rewards, grid = st.columns([3, 2])
    if update.steps:
        rewards.plotly_chart(live_reward_chart(update.steps), width="stretch", key="rl.live")
    if update.preview is not None and update.layout is not None:
        grid.plotly_chart(
            policy_chart(update.layout, update.preview), width="stretch", key="rl.live_policy"
        )


def _result(result: ReinforcementResult) -> None:
    """O treino terminado: as métricas, a recompensa, a política e a tabela Q."""
    st.markdown(
        f"**{result.agent_display_name}** em **{result.environment_name}** · seed "
        f"{result.seed} · run `{result.run_id}`"
    )
    st.caption(
        f"Treinou {format_count(result.steps)} episódios "
        f"({format_number(result.duration_seconds, 2)} s, com a avaliação)."
    )
    render_notices(result.notices, notes_title="Notas do treino")
    _metrics(result)
    st.subheader("Recompensa por episódio")
    st.plotly_chart(reward_chart(result.rewards), width="stretch", key="rl.rewards")
    st.caption(explain("reward_curve").text)
    st.subheader("O que o agente aprendeu")
    frame = _chosen_frame(result)
    st.plotly_chart(policy_chart(result.layout, frame), width="stretch", key="rl.policy")
    st.caption(explain("policy_map").text)
    st.markdown("**A tabela Q, ação a ação**")
    st.plotly_chart(q_table_chart(result.layout, frame), width="stretch", key="rl.q_table")
    _save_button(result)


def _metrics(result: ReinforcementResult) -> None:
    learned = result.reports[Partition.TEST]
    training = result.reports[Partition.TRAIN]
    columns = st.columns(4)
    cards = (
        ("Sucesso com a política aprendida", learned.metric("success_rate")),
        ("Sucesso a treinar", training.metric("success_rate")),
        ("Recompensa média (teste)", learned.metric("mean_reward")),
        ("Passos por episódio (teste)", learned.metric("mean_length")),
    )
    for column, (label, metric) in zip(columns, cards, strict=True):
        column.metric(label, metric.formatted, help=f"{metric.what} {metric.how_to_read}")


def _chosen_frame(result: ReinforcementResult) -> PolicyFrame:
    """A fotografia escolhida no slider (ou a do fim, se o run não tiver as intermédias)."""
    frames = result.trace.frames if result.trace is not None else ()
    if len(frames) < 2:
        if result.trace is None:
            st.caption(
                "Este run abriu do histórico sem os episódios intermédios: só ficam nos runs "
                "guardados («Guardar experiência»). Vês o que o agente sabia no fim."
            )
        return result.learned
    episodes = [frame.episode for frame in frames]
    episode = st.select_slider(
        "Episódio",
        options=episodes,
        value=episodes[-1],
        key=f"rl.episode.{result.run_id}",
    )
    return frames[episodes.index(episode)]


def _save_button(result: ReinforcementResult) -> None:
    """Guardar o run: junta ao histórico o agente e o que ele sabia em cada snapshot."""
    try:
        saved = get_lab().history.entry(result.run_id).saved
    except MLLabError:  # o run não ficou no histórico (por exemplo, foi apagado)
        return
    if st.button(
        "Experiência guardada" if saved else "Guardar experiência",
        icon=":material/check:" if saved else ":material/save:",
        disabled=saved,
        key="rl.save",
        help=(
            "Cada treino já fica no histórico (ecrã Experiências). Guardar junta-lhe o agente e "
            "os episódios intermédios: volta a abrir com tudo, mesmo depois de fechar a app."
        ),
    ):
        try:
            get_lab().history.save(result.run_id)
        except MLLabError as error:
            render_error(error)
        else:
            st.toast("Experiência guardada: está no ecrã Experiências.", icon=":material/save:")
            st.rerun()


main()
