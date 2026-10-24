"""Ecrã 4 · Modelo: escolher o algoritmo e os hiperparâmetros."""

from itertools import groupby

import streamlit as st

from components.forms import schema_form
from components.layout import render_header
from components.notices import render_error
from components.state import choose_model, current_preparation, get_lab, save_job
from ml_lab.core.services import ExperimentConfig, MLLabError, ModelCard, ModelSpec
from ml_lab.formatting import format_count


def main() -> None:
    """Desenha o ecrã."""
    render_header("modelo")
    preparation = current_preparation()
    if preparation is None:
        st.info(
            "Prepara primeiro os dados no ecrã Preparar: o catálogo de modelos depende do tipo "
            "de tarefa.",
            icon=":material/tune:",
        )
        st.page_link(
            "pages/preparar.py", label="Ir para o ecrã Preparar", icon=":material/arrow_forward:"
        )
        return
    lab = get_lab()
    summary = preparation.summary
    dataset = lab.datasets.summary(preparation.dataset_id)
    train = summary.partitions[0]
    st.markdown(
        f"**{dataset.name}** · {summary.task_type.label.lower()} · "
        f"{format_count(train.n_rows)} linhas de treino · {len(summary.feature_names)} features"
    )
    cards = lab.catalog.models(summary.task_type)
    if not cards:
        st.warning(
            f"Ainda não há modelos para {summary.task_type.label.lower()}: chegam na fase 4.",
            icon=":material/construction:",
        )
        return

    _catalog(cards)
    by_name = {card.name: card for card in cards}
    names = list(by_name)
    suggested = lab.catalog.suggested_model(summary.task_type)
    name = st.selectbox(
        "Modelo a treinar",
        names,
        index=names.index(suggested) if suggested in names else 0,
        format_func=lambda option: by_name[option].display_name,
        key=f"modelo.{summary.task_type}",
        persist_state="session",
        help="Começa pelo sugerido: é a referência que os modelos mais complexos têm de bater.",
    )
    card = by_name[name]
    choose_model(card)
    with st.expander("Ficha didática", icon=":material/menu_book:"):
        st.markdown(card.doc or "A ficha didática deste modelo ainda não foi escrita.")

    st.subheader("Hiperparâmetros")
    values = schema_form(card.hyperparameters_schema, key=f"modelo.{name}")
    st.caption(
        f"Seed da experiência: {preparation.seed}, a mesma da divisão dos dados. O modelo recebe "
        "uma seed derivada dela, por isso o resultado repete-se."
    )
    if st.button("Treinar", type="primary", icon=":material/play_arrow:", key="modelo.train"):
        config = ExperimentConfig(
            dataset=dataset.source,
            preparation=preparation.config,
            model=ModelSpec(name=name, hyperparameters=values),
            seed=preparation.seed,
        )
        try:
            job = lab.experiments.start(config)
        except MLLabError as error:  # configuração inválida: o treino nem chega a começar
            render_error(error)
            return
        save_job(job)
        st.switch_page("pages/treinar.py")


def _catalog(cards: list[ModelCard]) -> None:
    for family, members in groupby(cards, key=lambda card: card.family):
        st.markdown(f"**{family.label}**")
        group = list(members)
        for column, card in zip(st.columns(min(3, len(group))), group, strict=False):
            with column.container(border=True):
                st.markdown(f"**{card.display_name}**")
                badges = [card.implementation.label]
                if card.supports_proba:
                    badges.append("probabilidades")
                if card.iterative:
                    badges.append("treino passo a passo")
                st.caption(" · ".join(badges))
                st.write(card.summary)


main()
