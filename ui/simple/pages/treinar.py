"""Ecrã 6 · Treinar: acompanhar o treino a decorrer e revê-lo passo a passo."""

import streamlit as st

from charts.network import activations_chart, gradients_chart, network_chart, stuck_chart
from charts.training import animation_chart, embedding_chart, loss_chart
from components.layout import render_header, render_why
from components.notices import render_error, render_notices
from components.state import current_job, get_lab, last_result
from ml_lab.core.services import (
    ExperimentResult,
    JobStatus,
    Partition,
    TrainingAnimation,
    TrainingJob,
    explain,
)
from ml_lab.formatting import format_count, format_number, format_significant

# De quanto em quanto tempo (em segundos) o ecrã volta a ler o progresso do treino.
_REFRESH_SECONDS = 0.5


def main() -> None:
    """Desenha o ecrã."""
    render_header("treinar")
    job = current_job()
    if job is not None and not job.done:
        _live(job)
        return
    if job is not None and job.error is not None:
        render_error(job.error)
        st.page_link("pages/modelo.py", label="Voltar ao ecrã Modelo", icon=":material/arrow_back:")
        return
    result = last_result()
    if result is None:
        st.info(
            "Ainda não treinaste nenhum modelo: escolhe-o no ecrã Modelo e carrega em Treinar.",
            icon=":material/model_training:",
        )
        st.page_link(
            "pages/modelo.py", label="Ir para o ecrã Modelo", icon=":material/arrow_forward:"
        )
        return
    _replay(result)
    if result.network is not None:
        _network(result)
        render_why("neural_network", "backpropagation", "early_stopping", "loss", "snapshots")
    else:
        render_why("loss", "gradient_descent", "snapshots", "decision_boundary")


@st.fragment(run_every=_REFRESH_SECONDS)
def _live(job: TrainingJob[ExperimentResult]) -> None:
    """O treino a decorrer: só esta parte do ecrã se atualiza, a cada meio segundo."""
    if job.done:
        st.rerun()  # o ecrã inteiro volta a correr e mostra o resultado
    # O mesmo pedido que a API transmite ao frontend: o estado, os passos e a fronteira.
    update = get_lab().experiments.updates(job.run_id)
    state = update.state
    pause, stop, _ = st.columns([1, 1, 4])
    if state.status is JobStatus.PAUSED:
        pause.button(
            "Retomar",
            icon=":material/play_arrow:",
            on_click=job.resume,
            key="treinar.resume",
            width="stretch",
        )
    else:
        pause.button(
            "Pausar",
            icon=":material/pause:",
            on_click=job.pause,
            key="treinar.pause",
            width="stretch",
        )
    stop.button(
        "Parar",
        icon=":material/stop:",
        on_click=job.cancel,
        key="treinar.stop",
        width="stretch",
        help=(
            "O treino pára no fim do passo atual; o modelo fica com o que já aprendeu e é "
            "avaliado na mesma."
        ),
    )
    progress, planned = state.progress, state.planned_steps
    text = f"{state.status.label} · passo {format_count(progress.steps)}"
    if planned:
        text += f" de {format_count(planned)}, no máximo"
    if progress.last_loss is not None:
        text += f" · loss {format_significant(progress.last_loss)}"
    if planned:
        st.progress(min(1.0, progress.steps / planned), text=text)
    else:
        st.markdown(text)
    boundary, curve = st.columns([3, 2])
    if update.preview is not None:
        boundary.plotly_chart(
            animation_chart(update.preview), width="stretch", key="treinar.preview"
        )
    if update.steps:
        curve.plotly_chart(loss_chart(update.steps), width="stretch", key="treinar.live_loss")


def _replay(result: ExperimentResult) -> None:
    """O treino terminado: a animação dos snapshots e a curva da loss."""
    st.markdown(
        f"**{result.model_display_name}** em **{result.dataset_name}** · "
        f"{result.task_type.label.lower()} · seed {result.seed} · run `{result.run_id}`"
    )
    if result.steps:
        st.caption(
            f"Treinou em {format_count(result.steps)} passos "
            f"({format_number(result.duration_seconds, 2)} s, com a avaliação)."
        )
    else:
        st.caption("Este modelo treina de uma vez (não há passos para rever): vês o modelo final.")
    render_notices(result.training_notices, notes_title="Notas do treino")
    animation = result.animation
    if result.embedding is not None:  # redução de dimensionalidade: o resultado é o mapa
        st.plotly_chart(embedding_chart(result.embedding), width="stretch", key="treinar.map")
        st.caption(
            "Cada ponto é uma linha de treino nas novas coordenadas. Se escolheste uma coluna de "
            "referência no ecrã Preparar, as cores são as suas classes: classes bem separadas "
            "querem dizer que a redução guardou o que as distingue."
        )
    elif animation is not None:
        st.plotly_chart(animation_chart(animation), width="stretch", key="treinar.animation")
        st.caption(_how_to_read(animation))
    else:
        st.info(_without_animation(result), icon=":material/visibility_off:")
    if result.history:
        train = result.reports[Partition.TRAIN]
        percent = {
            metric.name: metric.display_name for metric in train.metrics if metric.as_percent
        }
        st.subheader("Loss ao longo do treino")
        st.plotly_chart(
            loss_chart(result.history, percent_metrics=percent),
            width="stretch",
            key="treinar.loss",
        )
    st.page_link("pages/avaliar.py", label="Ver a avaliação", icon=":material/analytics:")


def _network(result: ExperimentResult) -> None:
    """As camadas da rede: o diagrama, os gradientes, os neurónios parados e as ativações."""
    trace = result.network
    assert trace is not None  # quem chama confirmou que é uma rede
    st.subheader("As camadas da rede")
    steps = [frame.step for frame in trace.frames]
    step = st.select_slider(
        "Passo do treino",
        options=steps,
        value=steps[-1],
        key=f"treinar.network_step.{result.run_id}",
    )
    index = steps.index(step)
    st.plotly_chart(network_chart(trace, index), width="stretch", key="treinar.network")
    st.caption(explain("network_diagram").text)
    gradients, stuck = st.columns(2)
    gradients.plotly_chart(gradients_chart(trace), width="stretch", key="treinar.gradients")
    gradients.caption(explain("network_gradients").text)
    stuck.plotly_chart(stuck_chart(trace), width="stretch", key="treinar.stuck")
    stuck.caption(explain("network_stuck").text)
    st.plotly_chart(activations_chart(trace, index), width="stretch", key="treinar.activations")
    st.caption(explain("network_activations").text)


def _without_animation(result: ExperimentResult) -> str:
    """Porque é que não há animação: o run veio do histórico sem ser guardado, ou não há plano."""
    job = current_job()
    if job is None or job.run_id != result.run_id:  # abriu-se do histórico
        return (
            "Este run abriu do histórico sem o treino passo a passo (a animação e, numa rede, "
            "as camadas): só fica nos runs guardados («Guardar experiência»). Para o voltar a "
            "ver, reproduz o run no ecrã Experiências."
        )
    return (
        "Não há como desenhar o modelo: os dados não têm uma projeção 2D (escolhe uma no ecrã "
        "Preparar)."
    )


def _how_to_read(animation: TrainingAnimation) -> str:
    replay = (
        " Carrega em «▶ Reproduzir» para ver o treino desde o primeiro passo, ou arrasta o slider."
        if animation.animated
        else ""
    )
    if animation.value_kind == "probability" and animation.dimensions == 2:
        return (
            "As cores são a probabilidade que o modelo dá a cada classe em cada ponto do plano; a "
            "linha preta é a fronteira de decisão (50%). Os pontos são o treino." + replay
        )
    if animation.value_kind == "probability":
        return (
            "A curva é a probabilidade da classe positiva ao longo da feature; a fronteira de "
            "decisão é onde a curva corta os 50% (a linha tracejada)." + replay
        )
    if animation.value_kind == "class" and animation.field_labels != animation.class_labels:
        noise = (
            " As zonas cinzentas são ruído (não pertencem a nenhum grupo)."
            if animation.noise
            else ""
        )
        return (
            "Cada cor é o grupo que o modelo daria a um ponto nessa zona do plano; as cruzes são "
            "os centros e as linhas pontilhadas as elipses das nuvens, se o modelo os tiver. Os "
            "pontos são o treino, com o símbolo da coluna de referência (se a escolheste)."
            + noise
            + replay
        )
    if animation.value_kind == "class":
        return (
            "Cada cor é a classe que o modelo prevê nessa zona do plano: as fronteiras de decisão "
            "são as mudanças de cor. Os pontos são o treino." + replay
        )
    if animation.value_kind == "score":
        return (
            "As cores são a pontuação de anomalia em cada ponto do plano: quanto mais escuro, mais "
            "estranho seria um ponto ali. Os pontos são o treino." + replay
        )
    if animation.dimensions == 1:
        return "A linha é a previsão do modelo; os pontos são o treino." + replay
    return (
        "As cores são o valor que o modelo prevê em cada ponto do plano; os pontos são o treino, "
        "pintados com o valor real na mesma escala." + replay
    )


main()
