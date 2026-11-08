"""As ferramentas do analista: o que ele pode ler de uma procura acabada (fase 13, ADR 0053).

O analista é um modelo de linguagem que responde a perguntas sobre uma procura ("Why did the
winner win?"). Não vê a procura: sabe só o que as ferramentas lhe dão, quando as pede. Cada
ferramenta lê o resultado e a conversa dos agentes (``SearchFacts``) e devolve um JSON curto:

- ``list_candidates``: o quadro inteiro, do melhor para o pior (um candidato por linha), com
  cada modelo que entrou e o melhor candidato dele;
- ``get_candidate``: um candidato em pormenor (os folds, o treino, quem o propôs e porquê, e o
  que o agente dele disse);
- ``compare_candidates``: dois lado a lado (quem ganha e por quanto, se é um empate na prática,
  quantos folds ganhou cada um);
- ``data_plan``: o que o agente de dados decidiu, as linhas e os modelos que entraram.

**As contas fazem-se aqui, e não no modelo de linguagem.** A distância entre dois candidatos, o
empate dentro do desvio padrão, os folds ganhos e o "decora o treino" vêm prontos, escritos como o
ecrã os mostra. Um modelo de linguagem engana-se nas contas e lê mal uma diferença de pontos; com
as leituras feitas, só tem de as explicar. E as ferramentas não medem nada (não treinam nem olham
para os dados): só leem o que a procura já mediu, e há uma regra de arquitetura que o confirma.

Um pedido que não serve (um candidato que não existe, um argumento que falta) não é um erro da
pergunta: a ferramenta responde com a razão, o analista lê-a e pode corrigir-se.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ml_lab.core.automl.data_agent import DataDecision
from ml_lab.core.automl.model_agent import SkippedModel
from ml_lab.core.automl.objective import Objective
from ml_lab.core.automl.search_log import (
    AgentMessage,
    Candidate,
    MessageKind,
    model_agent_name,
)
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.llm import Tool, ToolCall
from ml_lab.formatting import format_count, format_number, format_ordinal, format_quantity

MAX_ROWS: Final = 40
"""Quantas linhas o quadro tem, no máximo. A conversa inteira do analista tem de caber nos 8192
tokens do Ollama (``CONTEXT_LENGTH``), e mediu-se com o ``qwen3:8b`` numa procura do Titanic: as
instruções e as ferramentas custam uns 970 tokens, cada linha uns 65, e o resto do quadro (o
vencedor, a referência e os modelos) uns 1100. Com 40 linhas, a conversa começa perto dos 4900, e
ainda cabem duas ferramentas de pormenor e o raciocínio. Com o orçamento por omissão (24
candidatos) e até 40, o quadro leva todos. Acima disso (o orçamento vai até 200), os de baixo
ficam de fora das linhas, mas nunca dos modelos: o melhor de cada modelo está sempre lá, e o
``get_candidate`` lê qualquer um pelo número."""

DEFAULT_SETTINGS: Final = "its default settings"
"""Como se diz que um candidato não mudou nenhum hiperparâmetro."""

MIN_SECONDS: Final = 0.1
"""Abaixo disto, o tempo de um candidato escreve-se "under 0.1 s" (e não "0.0 s")."""

NOT_TESTED: Final = "Not tested: only the winner goes to the test, so it has no test score."
"""O que se diz do teste de um candidato que não é o vencedor. Sem isto, o modelo de linguagem
inventava a nota de teste da referência e do segundo, para os comparar com o vencedor."""


@dataclass(frozen=True, slots=True)
class SearchFacts:
    """O que o analista pode ler de uma procura acabada: o resultado e a conversa dos agentes.

    O serviço monta-o a partir do resultado (``AutoMLResult``). Tem só o que o ecrã da procura já
    mostra: do teste, só a nota do vencedor, a única vez que o teste se usou.

    Attributes:
        dataset: O nome a mostrar do dataset.
        target: A coluna a prever.
        task_type: A tarefa.
        objective: A métrica que os agentes otimizaram.
        candidates: Todos os candidatos, pela ordem em que foram medidos.
        ranking: Os números dos candidatos medidos, do melhor para o pior.
        winner: O número do vencedor, se houver.
        baseline: O número da referência (o modelo sugerido, com os valores por omissão).
        messages: A conversa dos agentes, do princípio ao fim.
        settings: Para cada candidato, os hiperparâmetros que mudou, por palavras e com o valor
            por omissão ao lado ("maximum depth 8 (default 5)").
        test: A nota do vencedor no teste, escrita como o ecrã a mostra (``None`` sem teste).
        decisions: As decisões do agente de dados, com a razão de cada uma.
        adopted: Se o agente de dados adotou uma preparação feita à mão.
        train_rows: As linhas de treino.
        test_rows: As linhas de teste.
        search_rows: As linhas que a validação cruzada usou.
        folds: Os folds da validação cruzada.
        budget: O orçamento, em candidatos.
        finalists: Quantos modelos passaram da triagem à afinação.
        agents: Os nomes dos agentes de modelo chamados.
        skipped: Os modelos que ficaram de fora, com a razão.
        ai_model: O modelo de linguagem do agente de IA, se ele entrou na afinação.
        stopped_early: Se a procura parou a pedido (sem vencedor treinado nem teste).
        readings: A leitura do resultado, como o ecrã a mostra (os textos dos avisos).
    """

    dataset: str
    target: str
    task_type: TaskType
    objective: Objective
    candidates: tuple[Candidate, ...]
    ranking: tuple[int, ...]
    winner: int | None
    baseline: int | None
    messages: tuple[AgentMessage, ...]
    settings: Mapping[int, str]
    test: str | None
    decisions: tuple[DataDecision, ...]
    adopted: bool
    train_rows: int
    test_rows: int
    search_rows: int
    folds: int
    budget: int
    finalists: int
    agents: tuple[str, ...]
    skipped: tuple[SkippedModel, ...]
    ai_model: str | None = None
    stopped_early: bool = False
    readings: tuple[str, ...] = ()

    def candidate(self, number: int) -> Candidate:
        """O candidato ``number``.

        Raises:
            ToolRefusedError: Se a procura não tiver esse candidato.
        """
        if not 1 <= number <= len(self.candidates):
            where = (
                f"it has candidates #1 to #{len(self.candidates)}"
                if self.candidates
                else "it measured no candidate"
            )
            raise ToolRefusedError(f"There's no candidate #{number} in this search: {where}.")
        return self.candidates[number - 1]

    def rank(self, number: int) -> int | None:
        """A posição do candidato no quadro (1 é o vencedor), ou ``None`` se falhou."""
        return self.ranking.index(number) + 1 if number in self.ranking else None


class ToolRefusedError(Exception):
    """Uma ferramenta não pode responder ao que se pediu, e diz porquê.

    Nunca sai do analista: a razão volta ao modelo de linguagem como o resultado da ferramenta.
    """


@dataclass(frozen=True, slots=True)
class ToolUse:
    """Uma ferramenta que o analista usou: o que pediu e o que leu.

    Attributes:
        name: O nome da ferramenta (``"get_candidate"``).
        label: O nome a mostrar ("One candidate").
        arguments: Os argumentos, como o modelo de linguagem os escreveu.
        summary: O que devolveu, numa frase, em inglês.
        result: O que devolveu, tal como o modelo de linguagem o leu (JSON).
        error: Se a ferramenta recusou o pedido, porquê (o modelo também leu a razão).
    """

    name: str
    label: str
    arguments: dict[str, Any]
    summary: str
    result: str
    error: str | None = None


type Reader = Callable[[SearchFacts, Mapping[str, Any]], tuple[str, dict[str, Any]]]
"""O que uma ferramenta faz: lê a procura com os argumentos e devolve o resumo e o resultado."""


@dataclass(frozen=True, slots=True)
class AnalystTool:
    """Uma ferramenta do analista.

    Attributes:
        tool: O que o modelo de linguagem vê: o nome, a descrição e os argumentos.
        label: O nome a mostrar.
        purpose: O que faz, para quem usa o laboratório (o painel do analista mostra-o).
        read: O que faz.
    """

    tool: Tool
    label: str
    purpose: str
    read: Reader

    @property
    def name(self) -> str:
        """O nome da ferramenta."""
        return self.tool.name


# --- o quadro ------------------------------------------------------------------------


def _board(facts: SearchFacts, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """O quadro: o índice da procura, com o vencedor à parte, cada modelo e cada candidato.

    Leva **todos** os candidatos (até ``MAX_ROWS``), e não só os primeiros: o analista tem de ver
    a procura inteira para dizer porque ganhou o vencedor, e um modelo que ficou em 15.º também
    conta a história. Não há argumento para pedir menos (um ``limit`` que venha de uma resposta
    guardada da etapa 13.3 não conta): com ele, o ``qwen3:8b`` pedia sempre os dez primeiros.

    As lições do ``qwen3:8b`` a sério desenharam-no (ADR 0053 e 0054):

    - num quadro com um objeto de JSON por candidato (oito linhas cada um), um modelo pequeno
      troca as linhas (atribui a um candidato os números de outro) e faz as contas que não lhe
      pedem. Por isso o vencedor vem à parte, e **cada candidato é uma linha só**, com o número,
      o modelo e a nota juntos e já a dizer quanto fica atrás do primeiro. Uma linha custa também
      menos tokens do que o objeto (uns 65 contra uns 100, no Titanic), e é isso que deixa caber
      o quadro inteiro (``MAX_ROWS``);
    - para "que modelos entraram?" e "como se saiu o KNN?", ler vinte linhas é pedir um erro:
      ``models`` diz cada modelo uma vez, com quantos candidatos teve e o melhor deles;
    - com tudo no quadro (o treino, o empate, os folds), o modelo respondia só com ele, e lia mal
      o que lá estava. O quadro é por isso um índice, e os pormenores ficam nas outras
      ferramentas, cada uma com o seu papel: o treino e cada fold num candidato, o empate e os
      folds ganhos em dois lado a lado. O vencedor traz as chamadas seguintes já escritas.
    """
    ranked = [facts.candidate(number) for number in facts.ranking]
    shown = ranked[:MAX_ROWS]
    first = ranked[0] if ranked else None
    result: dict[str, Any] = {
        "objective": _objective_text(facts),
        "candidates_measured": len(facts.candidates),
    }
    if facts.winner is not None:
        result["winner"] = _winner(facts, facts.candidate(facts.winner), ranked)
    elif first is not None:
        result["best_so_far"] = {
            **_brief(facts, first),
            "note": "The search stopped before a winner was trained and tested.",
        }
    if facts.baseline is not None:
        baseline = facts.candidate(facts.baseline)
        if baseline.number == facts.winner:
            result["baseline"] = (
                f"#{baseline.number} {baseline.model_display_name}, with its default settings, "
                "is the winner itself: no other candidate beat the baseline."
            )
        else:
            # O modelo de linguagem tende a comparar o teste do vencedor com um "teste" da
            # referência, que não existe: diz-se que ela não foi ao teste.
            result["baseline"] = {**_brief(facts, baseline, first), "test": NOT_TESTED}
    models = _models(facts, first)
    if models:
        result["models"] = models
    if shown:
        result["board"] = [
            _row(facts, candidate, rank, first) for rank, candidate in enumerate(shown, start=1)
        ]
    if len(ranked) > len(shown):
        result["not_shown"] = (
            f"{format_quantity(len(ranked) - len(shown), 'more candidate', 'more candidates')} "
            "below these, the weakest ones: each model's best is in models, and "
            "get_candidate(number=…) reads any candidate."
        )
    failed = [candidate for candidate in facts.candidates if candidate.failed]
    if failed:
        result["failed"] = [
            {"number": c.number, "model": c.model_display_name, "error": c.error} for c in failed
        ]
    if facts.readings:
        result["readings"] = list(facts.readings)
    # Os modelos com nota: um modelo em que falharam todos não tem lugar no quadro.
    count = format_quantity(len({c.model for c in ranked}), "model", "models")
    if not ranked:
        summary = "No candidate got a score: there's no board."
    elif len(shown) == len(ranked):
        summary = (
            f"All {format_quantity(len(ranked), 'candidate', 'candidates')}, best first, "
            f"from {count}."
        )
    else:
        summary = (
            f"The best {format_count(len(shown))} of {format_count(len(ranked))} candidates, "
            f"and the best of each of the {count}."
        )
    return summary, result


def _winner(facts: SearchFacts, winner: Candidate, ranked: Sequence[Candidate]) -> dict[str, Any]:
    """O vencedor, à parte: a nota, o segundo, o teste e as chamadas para saber mais."""
    found = _brief(facts, winner)
    second = ranked[1] if len(ranked) > 1 else None
    see_also = [
        f"get_candidate(number={winner.number}) has each fold, the training score and whether "
        "it memorises the training data."
    ]
    if second is not None:
        found["second"] = {**_brief(facts, second, winner), "test": NOT_TESTED}
        # As chamadas seguintes, já escritas: um modelo pequeno segue melhor um exemplo do que
        # uma regra (sem elas, respondia só com o quadro).
        see_also.insert(
            0,
            f"compare_candidates(first={winner.number}, second={second.number}) says why it beat "
            "the second: the gap, whether it's a tie in practice and the folds each one won.",
        )
    found["see_also"] = see_also
    if facts.test is not None:
        found["test"] = (
            f"{facts.test}. Only the winner goes to the test, once: it can't be compared with the "
            "other candidates' cross-validation scores."
        )
        reading = _test_reading(facts, winner.number)
        if reading is not None:
            found["test_reading"] = reading
    return found


def _brief(
    facts: SearchFacts, candidate: Candidate, first: Candidate | None = None
) -> dict[str, Any]:
    """Um candidato em poucas linhas: o número, o modelo, as mudanças e a nota."""
    brief: dict[str, Any] = {
        "number": candidate.number,
        "model": candidate.model_display_name,
        "settings": facts.settings.get(candidate.number, DEFAULT_SETTINGS),
        "validation": _score(facts, candidate),
    }
    behind = _behind(facts, candidate, first)
    if behind is not None:
        brief["behind_the_first"] = behind
    return brief


def _row(facts: SearchFacts, candidate: Candidate, rank: int, first: Candidate | None) -> str:
    """Uma linha do quadro: o lugar, o número, o modelo e a nota, e depois o que mudou e a ronda.

    ``"2nd: #3 Decision tree; 92.0% ± 2.0%, 1.0 points behind the first; its default settings;
    Round 1 · screening"``. As partes separam-se com ponto e vírgula, porque as mudanças já levam
    vírgulas ("number of trees 200 (default 100), maximum depth 8 (default none)").
    """
    score = _score(facts, candidate)
    behind = _behind(facts, candidate, first)
    parts = [
        f"{format_ordinal(rank)}: #{candidate.number} {candidate.model_display_name}",
        score if behind is None else f"{score}, {behind} behind the first",
        facts.settings.get(candidate.number, DEFAULT_SETTINGS),
        candidate.phase.label,
    ]
    if candidate.ai:
        parts.append("proposed by the AI agent")
    return "; ".join(parts)


def _models(facts: SearchFacts, first: Candidate | None) -> list[str]:
    """Cada modelo que entrou na procura, numa linha: quantos candidatos teve e o melhor deles.

    ``"Decision tree: 2 candidates, the best is #4 with 93.0% ± 2.0%, the winner"``. Pela ordem do
    melhor de cada um no quadro; os modelos em que falharam todos ficam no fim.
    """
    groups: dict[str, list[Candidate]] = {}
    for candidate in facts.candidates:
        groups.setdefault(candidate.model, []).append(candidate)
    place = {number: index for index, number in enumerate(facts.ranking)}
    unranked = len(place)

    def best_place(members: Sequence[Candidate]) -> int:
        return min((place.get(member.number, unranked) for member in members), default=unranked)

    lines: list[str] = []
    for members in sorted(groups.values(), key=best_place):
        name = members[0].model_display_name
        count = format_quantity(len(members), "candidate", "candidates")
        measured = [member for member in members if member.number in place]
        if not measured:
            lines.append(f"{name}: {count}, {'it' if len(members) == 1 else 'all'} failed")
            continue
        best = min(measured, key=lambda member: place[member.number])
        which = f"#{best.number}" if len(members) == 1 else f"the best is #{best.number}"
        line = f"{name}: {count}, {which} with {_score(facts, best)}"
        behind = _behind(facts, best, first)
        if best.number == facts.winner:
            line += ", the winner"
        elif behind is not None:
            line += f", {behind} behind the first"
        if len(measured) < len(members):
            line += f" ({format_count(len(members) - len(measured))} failed)"
        lines.append(line)
    return lines


def _training(facts: SearchFacts, candidate: Candidate) -> dict[str, Any]:
    """A nota nos folds de treino e se o candidato decora o treino (a regra do ``Objective``)."""
    if candidate.train_mean is None or candidate.mean is None:
        return {}
    return {
        "training": facts.objective.written(candidate.train_mean),
        "memorises_training": facts.objective.overfits(candidate.train_mean, candidate.mean),
    }


def _behind(facts: SearchFacts, candidate: Candidate, first: Candidate | None) -> str | None:
    """Quanto fica atrás do primeiro do quadro (``None`` para o próprio primeiro)."""
    if first is None or first.number == candidate.number:
        return None
    if first.mean is None or candidate.mean is None:
        return None
    return facts.objective.gap(first.mean, candidate.mean)


def _test_reading(facts: SearchFacts, winner: int) -> str | None:
    """O que o coordenador disse do teste do vencedor, comparado com a validação cruzada."""
    readings = [
        message.text
        for message in facts.messages
        if message.candidate == winner
        and message.kind is MessageKind.TEST
        and not message.recipients
    ]
    return readings[-1] if readings else None


# --- um candidato --------------------------------------------------------------------


def _candidate(facts: SearchFacts, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    candidate = facts.candidate(_whole(arguments, "number"))
    objective = facts.objective
    rank = facts.rank(candidate.number)
    result: dict[str, Any] = {
        "number": candidate.number,
        "model": candidate.model_display_name,
        "family": candidate.family.label,
        "round": candidate.phase.label,
        "rank": "failed" if rank is None else f"{format_ordinal(rank)} of {len(facts.ranking)}",
        "winner": candidate.number == facts.winner,
        "baseline": candidate.number == facts.baseline,
        "settings": facts.settings.get(candidate.number, DEFAULT_SETTINGS),
        "hyperparameters": candidate.hyperparameters,
    }
    if candidate.failed or candidate.mean is None:
        result["failed"] = candidate.error
        summary = f"#{candidate.number} {candidate.model_display_name}: it failed."
    else:
        result["validation"] = _score(facts, candidate)
        result["folds"] = [objective.written(value) for value in candidate.folds]
        result.update(_training(facts, candidate))
        meaning = objective.reading(candidate.mean)
        if meaning:
            result["meaning"] = f"{meaning}."
        if candidate.number == facts.winner and facts.test is not None:
            result["test"] = (
                f"{facts.test}. Only the winner goes to the test, once: it can't be compared with "
                "the other candidates' cross-validation scores."
            )
            reading = _test_reading(facts, candidate.number)
            if reading is not None:
                result["test_reading"] = reading
        summary = f"#{candidate.number} {candidate.model_display_name}: {_score(facts, candidate)}."
    result["time"] = (
        "under 0.1 s for all the folds"
        if candidate.seconds < MIN_SECONDS
        else f"{format_number(candidate.seconds, 1)} s for all the folds"
    )
    result["proposed_by"] = "the AI agent" if candidate.ai else model_agent_name(candidate.family)
    if candidate.reason:
        result["reason_given"] = candidate.reason
    said = _said_about(facts, candidate.number)
    if said is not None:
        result["what_its_agent_said"] = said
    return summary, result


def _said_about(facts: SearchFacts, number: int) -> str | None:
    """O que o agente disse por último sobre o candidato: o resultado e o que quer dizer."""
    replies = [
        message.text
        for message in facts.messages
        if message.candidate == number and message.kind is MessageKind.REPLY
    ]
    return replies[-1] if replies else None


# --- dois candidatos -----------------------------------------------------------------


def _compare(facts: SearchFacts, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    first = facts.candidate(_whole(arguments, "first"))
    second = facts.candidate(_whole(arguments, "second"))
    if first.number == second.number:
        raise ToolRefusedError(
            f"Choose two different candidates: both numbers are #{first.number}. The board has "
            "the numbers."
        )
    means = {candidate.number: _measured(candidate) for candidate in (first, second)}
    objective = facts.objective
    # O melhor é o que está mais acima no quadro: o desempate (o desvio padrão, a ordem) é o
    # mesmo que escolheu o vencedor.
    better, worse = sorted((first, second), key=lambda c: facts.ranking.index(c.number))
    gap = objective.gap(means[better.number], means[worse.number])
    tied = _tied(better, worse)
    result: dict[str, Any] = {
        "first": _side(facts, first),
        "second": _side(facts, second),
        "better": f"#{better.number} ({better.model_display_name})",
        "gap": gap,
        "spread_between_folds": (
            f"{objective.written(first.std)} for #{first.number} and "
            f"{objective.written(second.std)} for #{second.number}"
        ),
        "tied_in_practice": tied,
    }
    tie_break = _tie_break(objective, better, worse)
    if tie_break is not None:
        result["why_ahead"] = f"#{better.number} is ahead because {tie_break}."
    won = _folds_won(objective, first, second)
    if won is not None:
        result["folds_won"] = {
            f"#{first.number}": won[0],
            f"#{second.number}": won[1],
            "level": won[2],
        }
        result["same_folds"] = (
            "Every candidate was measured on the same folds, so they can be compared fold by fold."
        )
    if first.model == second.model:
        names = sorted(set(first.hyperparameters) | set(second.hyperparameters))
        result["settings_that_differ"] = {
            name: {
                f"#{first.number}": first.hyperparameters.get(name),
                f"#{second.number}": second.hyperparameters.get(name),
            }
            for name in names
            if first.hyperparameters.get(name) != second.hyperparameters.get(name)
        }
    else:
        result["models"] = "Two different models: each one's settings are on its side."
    head = (
        f"#{first.number} {first.model_display_name} against #{second.number} "
        f"{second.model_display_name}"
    )
    if tied:
        summary = f"{head}: level in practice ({gap} apart, less than the spread between folds)."
    else:
        summary = f"{head}: #{better.number} is {gap} better."
    return summary, result


def _tied(better: Candidate, worse: Candidate) -> bool:
    """Se a distância é menor do que o desvio padrão do melhor entre folds: um empate na prática.

    É a regra do aviso ``close_race`` do resultado, para o quadro e a comparação dizerem o mesmo
    que o ecrã.
    """
    if better.mean is None or worse.mean is None:
        return False
    return abs(better.mean - worse.mean) < (better.std or 0.0)


def _tie_break(objective: Objective, better: Candidate, worse: Candidate) -> str | None:
    """Porque é que ``better`` fica à frente, quando as notas se escrevem iguais.

    É o desempate do coordenador (``rank``): a média com todas as casas decimais, depois o menor
    desvio padrão (o mais estável) e, por fim, o que se mediu primeiro. Com notas que se escrevem
    diferentes, não é preciso: a distância já o diz.
    """
    if better.mean is None or worse.mean is None:
        return None
    if objective.written(better.mean) != objective.written(worse.mean):
        return None
    if better.mean != worse.mean:
        return "its mean is a little better, in decimals the board doesn't show"
    if (better.std or 0.0) != (worse.std or 0.0):
        return (
            "the means are the same and its spread between folds is smaller: in a tie, the more "
            "stable one goes first"
        )
    return "the means and the spreads are the same, and it was measured first"


def _measured(candidate: Candidate) -> float:
    """A nota de um candidato na validação cruzada.

    Raises:
        ToolRefusedError: Se o candidato falhou (não tem nota para comparar).
    """
    if candidate.mean is None:
        raise ToolRefusedError(
            f"Candidate #{candidate.number} ({candidate.model_display_name}) failed, so it has "
            f"no score to compare: {candidate.error}"
        )
    return candidate.mean


def _folds_won(
    objective: Objective, first: Candidate, second: Candidate
) -> tuple[int, int, int] | None:
    """Quantos folds ganhou cada um, e em quantos ficaram iguais (como o ecrã os escreve).

    Os folds são os mesmos para todos os candidatos (a mesma divisão do treino), por isso
    comparam-se um a um. Dois valores que se escrevem iguais contam como um empate, como na
    conversa dos agentes: 81.2% contra 81.2% é um empate para quem lê.
    """
    pairs = [
        (one, other)
        for one, other in zip(first.folds, second.folds, strict=False)
        if one is not None and other is not None
    ]
    if not pairs:
        return None
    level = [objective.written(one) == objective.written(other) for one, other in pairs]
    ahead = [objective.is_better(one, other) for one, other in pairs]
    first_won = sum(a and not tie for a, tie in zip(ahead, level, strict=True))
    ties = sum(level)
    return first_won, len(pairs) - first_won - ties, ties


def _side(facts: SearchFacts, candidate: Candidate) -> dict[str, Any]:
    side: dict[str, Any] = {
        "number": candidate.number,
        "model": candidate.model_display_name,
        "rank": facts.rank(candidate.number),
        "settings": facts.settings.get(candidate.number, DEFAULT_SETTINGS),
        "validation": _score(facts, candidate),
    }
    side.update(_training(facts, candidate))
    return side


# --- o plano dos dados ---------------------------------------------------------------


def _data_plan(facts: SearchFacts, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    result: dict[str, Any] = {
        "dataset": facts.dataset,
        "target": facts.target,
        "task": facts.task_type.label,
        "objective": f"{facts.objective.label} ({facts.objective.direction})",
        "decisions": [
            {"decision": decision.title, "reason": decision.reason} for decision in facts.decisions
        ],
        "rows": {
            "training": facts.train_rows,
            "test": facts.test_rows,
            "used_by_the_cross_validation": facts.search_rows,
        },
        "cross_validation": (
            f"{facts.folds} folds, on the training data only: the test data was kept for the winner"
        ),
        "budget": (
            f"{format_quantity(facts.budget, 'candidate', 'candidates')}, with "
            f"{format_quantity(facts.finalists, 'finalist', 'finalists')} in the tuning"
        ),
        "agents": list(facts.agents),
    }
    if facts.adopted:
        result["preparation"] = "Made by hand in Prepare: the data agent adopted it as it was."
    if facts.search_rows < facts.train_rows:
        result["sample"] = (
            f"The cross-validation used a sample of {format_count(facts.search_rows)} of the "
            f"{format_count(facts.train_rows)} training rows; the winner trained on all of them."
        )
    if facts.ai_model is not None:
        result["ai_agent"] = f"Took part in the tuning, thinking with {facts.ai_model}."
    if facts.skipped:
        groups: dict[str, list[str]] = {}
        for model in facts.skipped:
            groups.setdefault(model.reason, []).append(model.display_name)
        result["left_out"] = [
            {"models": names, "reason": reason} for reason, names in groups.items()
        ]
    summary = (
        f"{format_quantity(len(facts.decisions), 'decision', 'decisions')} of the data agent, "
        f"{format_count(facts.train_rows)} training rows and {format_count(facts.test_rows)} "
        "test rows."
    )
    return summary, result


# --- as ferramentas ------------------------------------------------------------------

_NUMBER_ARGUMENT: Final = {
    "type": "integer",
    "minimum": 1,
    "description": "The candidate's number, the # on the board.",
}

TOOLS: Final[tuple[AnalystTool, ...]] = (
    AnalystTool(
        Tool(
            "list_candidates",
            "The search's whole board, an index: the winner (with its test score), the "
            "baseline, every model that took part (how many candidates it had and its best one) "
            "and every candidate, best first, one per line: its place, its number, the model, "
            "its cross-validation score (the mean across the folds ± the standard deviation), "
            "how far behind the first it is, the settings it changed and the round. The details "
            "are in the other tools: get_candidate (each fold, the training score, whether it "
            "memorises the training data) and compare_candidates (the gap, whether it's a tie "
            "in practice, the folds each one won). Start here.",
        ),
        "The candidate board",
        "Every candidate, best first, with its cross-validation score, and each model's best.",
        _board,
    ),
    AnalystTool(
        Tool(
            "get_candidate",
            "One candidate in detail: the model and the settings it changed (with the default "
            "values), the score in each fold, the mean and the spread, the score on the training "
            "folds and whether it memorises the training data, what the score means, who "
            "proposed it and why, and what its agent said after measuring it.",
            {"type": "object", "properties": {"number": _NUMBER_ARGUMENT}, "required": ["number"]},
        ),
        "One candidate",
        "A candidate in detail: each fold, the training score, who proposed it and why.",
        _candidate,
    ),
    AnalystTool(
        Tool(
            "compare_candidates",
            "Two candidates side by side: which one is better in the cross-validation and by how "
            "much, whether the gap is smaller than the spread between folds (a tie, in practice), "
            "how many folds each one won, their training scores and, for the same model, the "
            "settings that differ.",
            {
                "type": "object",
                "properties": {"first": _NUMBER_ARGUMENT, "second": _NUMBER_ARGUMENT},
                "required": ["first", "second"],
            },
        ),
        "Compare two candidates",
        "Two candidates side by side: the gap, the folds each one won, whether it's a tie.",
        _compare,
    ),
    AnalystTool(
        Tool(
            "data_plan",
            "What the data agent did before the search: each decision with its reason "
            "(cleaning, task, split, transformations, objective), the training, test and "
            "search rows, the cross-validation, the budget, and the models that took part or "
            "were left out (with why).",
        ),
        "The data plan",
        "What the data agent decided, the rows, and the models that took part.",
        _data_plan,
    ),
)
"""As ferramentas do analista, pela ordem em que se mostram."""

_BY_NAME: Final[Mapping[str, AnalystTool]] = {tool.name: tool for tool in TOOLS}


def use_tool(facts: SearchFacts, call: ToolCall) -> ToolUse:
    """Usa a ferramenta que o modelo de linguagem pediu, e guarda o que ele vai ler.

    Uma ferramenta que não existe, ou um pedido que não serve, dá uma resposta com a razão (no
    campo ``error``): o modelo lê-a como o resultado e pode corrigir o pedido.
    """
    arguments = dict(call.arguments)
    tool = _BY_NAME.get(call.name)
    if tool is None:
        names = ", ".join(_BY_NAME)
        problem = f"There's no tool called “{call.name}”. The tools are: {names}."
        return ToolUse(call.name, call.name, arguments, problem, _json({"error": problem}), problem)
    try:
        summary, result = tool.read(facts, arguments)
    except ToolRefusedError as refusal:
        problem = str(refusal)
        return ToolUse(
            tool.name, tool.label, arguments, problem, _json({"error": problem}), problem
        )
    return ToolUse(tool.name, tool.label, arguments, summary, _json(result))


def _json(data: Mapping[str, Any]) -> str:
    """O resultado como o modelo de linguagem o lê (e como o painel o mostra)."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _whole(arguments: Mapping[str, Any], name: str, *, default: int | None = None) -> int:
    """Um argumento inteiro: aceita o ``7.0`` e o ``"#7"`` que os modelos de linguagem escrevem.

    Raises:
        ToolRefusedError: Se faltar, ou não for um número inteiro.
    """
    value = arguments.get(name, default)
    if value is None:
        raise ToolRefusedError(
            f"“{name}” is missing: it's a candidate's number, the # on the board."
        )
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        value = int(text) if text.isdigit() else value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        written = json.dumps(value, default=str)
        raise ToolRefusedError(f"“{name}” must be a whole number, not {written}.")
    return value


def _score(facts: SearchFacts, candidate: Candidate) -> str:
    """A nota na validação cruzada, como o ecrã a mostra ("82.4% ± 1.9%")."""
    if candidate.mean is None:
        return "failed"
    objective = facts.objective
    return f"{objective.written(candidate.mean)} ± {objective.written(candidate.std)}"


def _objective_text(facts: SearchFacts) -> str:
    """O que mede a nota do quadro: a métrica, o sentido e os folds."""
    return (
        f"{facts.objective.label} ({facts.objective.direction}): the mean across "
        f"{facts.folds} cross-validation folds of the training data, ± the standard deviation "
        "between folds"
    )
