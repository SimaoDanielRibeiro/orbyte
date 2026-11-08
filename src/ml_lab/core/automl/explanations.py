"""Textos do painel "Porquê?" do ecrã Agentes (só se escrevem aqui, em inglês; ADR 0043)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

AUTOML_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "automl": Explanation(
            "The agents (AutoML)",
            "In the automatic mode, three kinds of agents do an analyst's job: the **data "
            "agent** prepares the table, an **agent per model family** proposes candidates (the "
            "trees, the linear models, the networks…) and the **coordinator** shares out the "
            "budget and picks the winner. Every decision stays in the conversation, with its "
            "reason. These agents have no magic and no language model: they're rules and "
            "measurements, and with the same seed the search repeats exactly. If Ollama is "
            "open, an **AI agent** that thinks with a language model can join them, and at the "
            "end an **analyst** answers your questions about the search.",
        ),
        "analyst": Explanation(
            "The analyst",
            "The **analyst** is an AI agent that answers questions about a finished search, such "
            "as “Why did the winner win?”. It starts with nothing but your question: it doesn't "
            "see the search. What it has are **tools**, and it chooses which to use: the "
            "**board** (the candidates, best first), **one candidate** in detail, **two "
            "candidates** side by side and the **data plan**. Its cycle is the one of any agent "
            "with tools: it **thinks**, **calls a tool**, **reads** what the tool returned and "
            "thinks again, until it knows enough to **answer**. You see every step: the "
            "reasoning, each tool with what it asked for and what it got back, and the answer.\n\n"
            "The tools do the maths (the gap between two candidates, whether it's smaller than the "
            "spread between folds, how many folds each one won), so the language model only has "
            "to explain them. And the rules check the answer: every number in it must be in what "
            "the tools returned. A language model can invent a number that sounds right; here, "
            "that shows. The analyst measures nothing new, and of the test it only sees what the "
            "result already shows: the winner's score.\n\n"
            "Its replies are kept with the answer: asking again with them gives the same answer, "
            "without Ollama.",
        ),
        "ai_agent": Explanation(
            "The AI agent",
            "The family agents draw the next combination of hyperparameters at random: they "
            "don't look at what's already been measured. The **AI agent** does. On its turn, it "
            "gets the finalists, each one's search space and the results of every candidate, "
            "**thinks** (its reasoning shows up live, in the conversation) and proposes the next "
            "candidate, with the reason. What thinks is a language model running on your "
            "computer, in Ollama.\n\n"
            "The rules around it keep the search honest: the answer has a fixed format, a "
            "proposal outside the search space (or repeated) is refused, and the coordinator is "
            "the one who measures, with cross-validation on the training data only. The AI agent "
            "never sees the test data nor picks the winner. At the end, the best it proposed is "
            "compared with the best of the random search, with the same budget.\n\n"
            "A language model doesn't always answer the same, not even with the same seed. So its "
            "answers are kept in the result's request: with them, the search repeats exactly, "
            "without Ollama.",
        ),
        "exchange_history": Explanation(
            "The exchange history",
            "It replays the search message by message, on the flowchart: each exchange is a "
            "light travelling from whoever speaks to whoever listens. The **data agent** hands "
            "the plan to the **coordinator**; the coordinator announces each round to the agents "
            "and gives the turn to one at a time (a **request**); the agent measures the "
            "candidate with cross-validation and replies with the result (the **reply**). At the "
            "end, the coordinator hands the best one to the **winner**, which trains on all the "
            "training data and goes to the **test**, just once. The timeline has a mark per "
            "message, in the colour of whoever said it: drag it to go to any moment of the "
            "search and see how it stood.",
        ),
        "automatic_or_manual": Explanation(
            "Automatic or manual?",
            "The automatic mode quickly answers “which model works best for this?” and shows "
            "why. The manual one (Explore, Prepare, Model, Train) is where you learn to make each "
            "choice and see the training from the inside. Both end in the same place: the "
            "agents' winner is a run like any other, evaluated and explained the same way, and "
            "any candidate on the board can be trained by hand.",
        ),
        "model_selection": Explanation(
            "Choosing without looking at the test",
            "The agents compare the candidates with **cross-validation**, on the training data "
            "only: the training data is split into folds, and each candidate trains on some and "
            "is evaluated on the one left out. The test data stays put away until the end and "
            "only the winner sees it, once. If the choice were made on the test data, the test "
            "would become just more training: the winner would be the luckiest on it, and its "
            "score would no longer say how it behaves on new data.",
        ),
        "screening_and_tuning": Explanation(
            "Screening and tuning",
            "Tuning every model would cost a lot. So there are two rounds: in **screening**, each "
            "model is tried once, with the default values (a cheap first impression); in "
            "**tuning**, only the best (the finalists) get the rest of the **budget**, so their "
            "agents can try other combinations of hyperparameters. The budget counts "
            "candidates, and each candidate trains once per fold.",
        ),
        "random_search": Explanation(
            "Random search",
            "In the tuning, the agents don't walk through a grid with every combination: they "
            "draw them at random, inside each model's search space (the SVM's C between 0.1 and "
            "100, on a logarithmic scale, for example). With the same number of tries, random "
            "search tries more different values of each hyperparameter, and that counts because, "
            "in each problem, only a few of them really matter.",
        ),
        "winners_curse": Explanation(
            "The winner is a bit optimistic",
            "Among many similar candidates, the one that wins the cross-validation is usually "
            "good and also had a bit of luck with the folds. So its test score usually lands a "
            "little below its cross-validation score, and the more candidates, the more so. "
            "Differences smaller than the standard deviation between folds are, in practice, "
            "ties: in those cases, the simpler model is as good a choice as the winner.",
        ),
        "train_candidate": Explanation(
            "Training a candidate by hand",
            "Any candidate on the board trains like a normal run, with the same preparation and "
            "the same seed: you see the training step by step and evaluate it on the test data. "
            "It's the bridge from the automatic mode to the manual one, for example to compare "
            "the winner with a simpler model that came in almost level.",
        ),
        "data_agent": Explanation(
            "What the data agent decides",
            "The data agent looks at the profile (not at the rows' values) and decides as in "
            "Prepare: it drops identifiers, free text and nearly empty columns, confirms the "
            "task, splits off 20% for the test (stratified, if there are classes), fills in the "
            "nulls, one-hot encodes the categories and standardises the numbers. With imbalanced "
            "classes, it suggests optimising the balanced accuracy. If you've already prepared "
            "the data by hand, it uses your preparation.",
        ),
    }
)
