"""Ecrã 2 · Agentes: o modo automático, que procura o melhor modelo para o que se quer prever.

O ecrã só monta o pedido e mostra o que os agentes dizem e encontram: as decisões, a
procura e a escolha do vencedor são todas do serviço (``lab.automl``). No fim, o vencedor
abre-se no Avaliar como qualquer run, e qualquer candidato se pode treinar à mão.
"""

import streamlit as st

from charts.automl import agents_chart, search_chart
from components.forms import schema_form
from components.layout import render_header, render_why, require_dataset
from components.notices import render_error, render_notices
from components.state import (
    current_automl_job,
    current_preparation,
    get_lab,
    last_automl_result,
    save_automl_job,
    save_job,
    show_result,
    training_in_progress,
)
from ml_lab.core.services import (
    AgentMessage,
    AgentRole,
    AutoMLPlan,
    AutoMLRequest,
    AutoMLResult,
    Candidate,
    DatasetSummary,
    JobStatus,
    MLLabError,
    ObjectiveOption,
    SearchSettings,
    TrainingJob,
    explain,
)
from ml_lab.formatting import format_count, format_number, format_percent, format_value

# De quanto em quanto tempo (em segundos) o ecrã volta a ler o progresso da procura.
_REFRESH_SECONDS = 0.5

# As mensagens mais recentes que a vista ao vivo mostra (a conversa toda fica no fim).
_LIVE_MESSAGES = 8

_AVATARS = {
    AgentRole.DATA: ":material/dataset:",
    AgentRole.COORDINATOR: ":material/hub:",
    AgentRole.MODEL: ":material/account_tree:",
}


def main() -> None:
    """Desenha o ecrã."""
    render_header("agentes")
    dataset = require_dataset()
    if dataset is None:
        return
    job = current_automl_job()
    if job is not None and not job.done:
        _live(job)
        return
    if job is not None and job.error is not None:
        render_error(job.error)
    result = last_automl_result()
    if result is not None and result.plan.dataset_id == dataset.id:
        _result(result)
        with st.expander("Fazer outro pedido aos agentes", icon=":material/groups:"):
            _setup(dataset)
    else:
        _setup(dataset)
    render_why(
        "automl",
        "automatic_or_manual",
        "data_agent",
        "model_selection",
        "screening_and_tuning",
        "random_search",
        "winners_curse",
    )


def _setup(dataset: DatasetSummary) -> None:
    """O pedido: o alvo, a métrica, os agentes e o orçamento; e o botão que começa a procura."""
    lab = get_lab()
    columns = [column.name for column in dataset.columns]
    suggested = dataset.suggested_target
    target = st.selectbox(
        "O que queres prever (a coluna-alvo)",
        columns,
        index=columns.index(suggested) if suggested in columns else len(columns) - 1,
        key=f"agentes.target.{dataset.id}",
    )
    suggestion = lab.datasets.suggest_task(dataset.id, target)
    tasks = [task for task in suggestion.allowed if lab.automl.options(task).supported]
    if suggestion.task_type is None or not tasks:
        reason = (
            suggestion.reason
            if suggestion.task_type is None
            else lab.automl.options(suggestion.task_type).reason
        )
        st.warning(reason, icon=":material/info:")
        st.page_link(
            "pages/preparar.py", label="Ir para o modo manual", icon=":material/arrow_forward:"
        )
        return
    task_column, objective_column = st.columns(2)
    task = task_column.selectbox(
        "Tarefa",
        tasks,
        format_func=lambda option: option.label,
        key=f"agentes.task.{dataset.id}.{target}",
        help=suggestion.reason,
    )
    try:
        suggested_plan = lab.automl.plan(
            AutoMLRequest(dataset=dataset.source, target=target, task_type=task, seed=42)
        )
    except MLLabError as error:  # o alvo não serve (por exemplo, tem um só valor)
        render_error(error)
        return
    options = lab.automl.options(task)
    objectives = {option.name: option for option in options.objectives}
    names = list(objectives)
    objective = objective_column.selectbox(
        "A métrica a otimizar",
        names,
        index=names.index(suggested_plan.objective.name),
        format_func=lambda name: objectives[name].label,
        key=f"agentes.objective.{dataset.id}.{target}.{task}",
        help="A sugerida é a do agente de dados (a razão está nas decisões dele, abaixo).",
    )
    cards = {card.family: card for card in options.agents}
    families = st.multiselect(
        "Os agentes chamados (um por família de modelos)",
        list(cards),
        default=list(cards),
        format_func=lambda family: family.label,
        key=f"agentes.families.{task}",
    )
    st.markdown("**O tamanho da procura**")
    settings = SearchSettings.model_validate(
        schema_form(options.settings_schema, key="agentes.settings")
    )
    seed = st.number_input(
        "Seed",
        min_value=0,
        value=42,
        step=1,
        key="agentes.seed",
        help="A mesma seed dá a mesma divisão, os mesmos folds e os mesmos sorteios da procura.",
    )
    preparation = current_preparation()
    own = (
        preparation is not None
        and preparation.dataset_id == dataset.id
        and preparation.config.target == target
        and st.checkbox(
            "Usar a minha preparação (do ecrã Preparar)",
            key="agentes.own_preparation",
            help="O agente de dados adota a tua limpeza, divisão e transformações.",
        )
    )
    request = AutoMLRequest(
        dataset=dataset.source,
        target=target,
        task_type=task,
        objective=objective,
        families=tuple(families),
        preparation=preparation.config if own and preparation is not None else None,
        settings=settings,
        seed=int(seed),
    )
    try:
        plan = lab.automl.plan(request)
    except MLLabError as error:
        render_error(error)
        return
    _plan(plan)
    if st.button(
        "Pedir aos agentes", type="primary", icon=":material/play_arrow:", key="agentes.start"
    ):
        try:
            job = lab.automl.start(request)
        except MLLabError as error:  # o pedido não serve: a procura nem chega a começar
            render_error(error)
            return
        save_automl_job(job)
        st.rerun()


def _plan(plan: AutoMLPlan) -> None:
    """O que os agentes vão fazer: os dados, as decisões do agente de dados e os agentes."""
    rows = (
        f"{format_count(plan.train_rows)} linhas de treino e {format_count(plan.test_rows)} de "
        "teste (guardadas até ao fim, para o vencedor)."
    )
    if plan.sampled:
        rows += (
            f" A validação cruzada usa uma amostra de {format_count(plan.search_rows)} linhas "
            "do treino, para a procura ser rápida."
        )
    st.caption(rows)
    title = "O que o agente de dados decidiu" if not plan.adopted else "A tua preparação"
    with st.expander(title, icon=":material/dataset:"):
        for decision in plan.decisions:
            st.markdown(f"**{decision.title}.** {decision.reason}")
    with st.expander("Os agentes de modelo e o que procuram", icon=":material/account_tree:"):
        for card in plan.agents:
            st.markdown(f"**{card.name}**")
            for model in card.models:
                searches = "; ".join(model.searches) or "só os valores por omissão"
                st.caption(f"{model.display_name} · {searches}")
        if plan.skipped:
            st.markdown("**Ficam de fora**")
            for skipped in plan.skipped:
                st.caption(f"{skipped.display_name}: {skipped.reason}.")


@st.fragment(run_every=_REFRESH_SECONDS)
def _live(job: TrainingJob[AutoMLResult]) -> None:
    """A procura a decorrer: só esta parte do ecrã se atualiza, a cada meio segundo."""
    if job.done:
        st.rerun()  # o ecrã inteiro volta a correr e mostra o resultado
    update = get_lab().automl.updates(job.run_id)
    state, objective = update.state, update.objective
    pause, stop, _ = st.columns([1, 1, 4])
    if state.status is JobStatus.PAUSED:
        pause.button(
            "Retomar", icon=":material/play_arrow:", on_click=job.resume, key="agentes.resume"
        )
    else:
        pause.button("Pausar", icon=":material/pause:", on_click=job.pause, key="agentes.pause")
    stop.button(
        "Parar",
        icon=":material/stop:",
        on_click=job.cancel,
        key="agentes.stop",
        help="A procura pára no fim do candidato atual; o vencedor é o melhor até aí.",
    )
    measured, planned = state.progress.steps, state.planned_steps
    text = f"{update.phase.label} · {format_count(measured)} candidatos medidos"
    if planned:
        st.progress(min(1.0, measured / planned), text=f"{text}, de {format_count(planned)}")
    else:
        st.markdown(text)
    if update.leader is not None:
        _leader(update.leader, objective)
    if update.candidates:
        st.plotly_chart(
            search_chart(update.candidates, objective), width="stretch", key="agentes.live"
        )
    for message in update.messages[-_LIVE_MESSAGES:]:
        _message(message)


def _leader(leader: Candidate, objective: ObjectiveOption) -> None:
    st.metric(
        "O melhor até agora, na validação cruzada",
        f"{_value(leader.mean, objective)} ± {_value(leader.std, objective)}",
        help=f"{leader.model_display_name} ({leader.family.label}).",
    )
    st.caption(f"{leader.model_display_name} · {_changes(leader)}")


def _result(result: AutoMLResult) -> None:
    """A procura terminada: o vencedor, a leitura, os gráficos, o quadro e a conversa."""
    plan, objective = result.plan, result.plan.objective
    st.markdown(
        f"**{plan.dataset_name}** · prever **{plan.target}** ({plan.task_type.label.lower()}) · "
        f"{objective.label} · seed {result.seed} · procura `{result.search_id}`"
    )
    st.caption(
        f"{format_count(len(result.candidates))} candidatos em "
        f"{format_number(result.duration_seconds, 1)} s, com o treino do vencedor."
    )
    render_notices(result.notices, notes_title="Leitura da procura")
    winner = result.winner_candidate
    if winner is not None:
        _winner(result, winner)
    st.subheader("A procura")
    st.plotly_chart(
        search_chart(result.candidates, objective, result.winner),
        width="stretch",
        key="agentes.search",
    )
    st.caption(explain("screening_and_tuning").text)
    st.subheader("O melhor de cada agente")
    baseline = None if result.baseline is None else result.candidate(result.baseline)
    st.plotly_chart(
        agents_chart(result.candidates, objective, baseline),
        width="stretch",
        key="agentes.agents",
    )
    st.subheader("O quadro dos candidatos")
    st.dataframe(_table(result), hide_index=True, width="stretch")
    _manual(result)
    with st.expander("A conversa dos agentes", icon=":material/forum:"):
        for message in result.messages:
            _message(message)


def _winner(result: AutoMLResult, winner: Candidate) -> None:
    objective = result.plan.objective
    st.markdown(f"**Vencedor: {winner.model_display_name}** · {winner.family.label}")
    st.caption(_changes(winner))
    validation, spread, test, reference = st.columns(4)
    validation.metric(
        "Validação cruzada",
        _value(winner.mean, objective),
        help="A média nos folds do treino: foi com isto que os agentes escolheram.",
    )
    spread.metric(
        "Desvio entre folds",
        f"± {_value(winner.std, objective)}",
        help="Quanto a nota muda de fold para fold. Diferenças menores do que isto entre dois "
        "candidatos são, na prática, empates.",
    )
    test.metric(
        "No teste (uma só vez)",
        result.test.formatted if result.test is not None else "—",
        help="O teste ficou guardado até ao fim: é a nota honesta do vencedor em dados novos.",
    )
    if result.baseline is not None:
        baseline = result.candidate(result.baseline)
        reference.metric(
            "Referência",
            _value(baseline.mean, objective),
            help=f"{baseline.model_display_name} com os valores por omissão, na validação "
            "cruzada: é o resultado que os outros tinham de bater.",
        )
    if result.run_id is not None and st.button(
        "Abrir o vencedor no Avaliar",
        type="primary",
        icon=":material/analytics:",
        key="agentes.open",
    ):
        _open(result.run_id)


def _open(run_id: str) -> None:
    """Abre o run do vencedor nos ecrãs do modo manual (Avaliar, Explicar)."""
    if training_in_progress():
        st.warning("Há um treino a decorrer no ecrã Treinar: espera que acabe, ou pára-o.")
        return
    try:
        show_result(get_lab().history.open(run_id))
    except MLLabError as error:
        render_error(error)
        return
    st.switch_page("pages/avaliar.py")


def _manual(result: AutoMLResult) -> None:
    """A ponte para o modo manual: treinar um candidato do quadro como um run normal."""
    if not result.ranking:
        return
    st.markdown("**Treinar um candidato à mão**")
    st.caption(explain("train_candidate").text)
    objective = result.plan.objective
    by_number = {candidate.number: candidate for candidate in result.candidates}
    number = st.selectbox(
        "Candidato",
        list(result.ranking),
        format_func=lambda n: (
            f"{result.ranking.index(n) + 1}.º · candidato {n} · {by_number[n].model_display_name}"
            f" · {_value(by_number[n].mean, objective)}"
        ),
        key=f"agentes.manual.{result.search_id}",
    )
    if st.button("Treinar no modo manual", icon=":material/model_training:", key="agentes.train"):
        try:
            job = get_lab().automl.train_candidate(result.search_id, number)
        except MLLabError as error:
            render_error(error)
            return
        save_job(job)
        st.switch_page("pages/treinar.py")


def _table(result: AutoMLResult) -> dict[str, list[object]]:
    """O quadro: os candidatos do melhor para o pior (os que falharam no fim)."""
    objective = result.plan.objective
    places = {number: place for place, number in enumerate(result.ranking, start=1)}
    ordered = sorted(result.candidates, key=lambda c: places.get(c.number, len(places) + c.number))
    return {
        "Lugar": [places.get(c.number, "—") for c in ordered],
        "Candidato": [c.number for c in ordered],
        "Agente": [c.family.label for c in ordered],
        "Modelo": [c.model_display_name for c in ordered],
        "Ronda": [c.phase.label for c in ordered],
        "O que mudou": [_changes(c) for c in ordered],
        objective.label: [_value(c.mean, objective) for c in ordered],
        "± (desvio)": [_value(c.std, objective) for c in ordered],
        "No treino": [_value(c.train_mean, objective) for c in ordered],
        "Segundos": [format_number(c.seconds, 2) for c in ordered],
        "Erro": [c.error or "" for c in ordered],
    }


def _message(message: AgentMessage) -> None:
    with st.chat_message(message.author, avatar=_AVATARS[message.role]):
        st.markdown(f"**{message.author}** · {message.phase.label}")
        st.markdown(message.text)


def _value(value: float | None, objective: ObjectiveOption) -> str:
    if value is None:
        return "—"
    return format_percent(value) if objective.as_percent else format_number(value, 3)


def _changes(candidate: Candidate) -> str:
    if not candidate.changes:
        return "valores por omissão"
    return ", ".join(f"{name} {format_value(value)}" for name, value in candidate.changes.items())


main()
