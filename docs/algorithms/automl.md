# AutoML agents (screening and random search)

## Intuition

Choosing a model by hand is always the same loop: prepare the data, try a model, measure, change a
hyperparameter, measure again. **AutoML** (automated machine learning) runs that loop on its own,
with a budget. In Orbyte, the work is split as in a team:

- the **data agent** prepares the table from the profile: it removes identifiers and free text,
  confirms the task, splits (20% for the test), fills in the nulls, one-hot encodes the categories
  and standardises the numbers. Each decision comes with its reason;
- an **agent per model family** (the linear models, the trees, the ensembles, the networks…)
  knows its family's models and what's worth trying in each one;
- the **coordinator** manages the budget and picks the winner.

The search has two rounds. In the **screening**, each model is tried once, with its default
values: a cheap first impression that shows which families get on well with this data. In the
**tuning**, only the best ones (the **finalists**) get the rest of the budget, and their agents
draw other combinations of hyperparameters.

There's no magic and no language model: it's rules and measurements. With the same seed, the
search repeats exactly, and the agents' conversation says why the winner won.

## Essential maths

Each candidate (a model with some hyperparameters) is measured with **$k$-fold cross-validation**,
on the training data only: the training data is split into $k$ parts, and the candidate trains $k$
times, each time on $k - 1$ parts, and is evaluated on the part left out. The score is the mean,
and the standard deviation says how much it varies from fold to fold:

$$\bar{m} = \frac{1}{k} \sum_{i=1}^{k} m_i \qquad s = \sqrt{\frac{1}{k} \sum_{i=1}^{k} (m_i - \bar{m})^2}$$

The winner is the candidate with the best mean (the highest, for accuracy; the lowest, for RMSE).
In a tie, the one with the smallest deviation wins, the most stable.

In the tuning, the hyperparameters are drawn from the **search space** each model declares (the
`x-search` of its hyperparameters). A number on a logarithmic scale, like the SVM's $C$ between
$0.1$ and $100$, is drawn in the exponents:

$$C = 10^{u}, \qquad u \sim \text{Uniform}(\log_{10} 0.1,\ \log_{10} 100)$$

That way, landing between $0.1$ and $1$ is as likely as landing between $10$ and $100$. That's
**random search**: with the same number of tries as a grid, it tries more different values of
each hyperparameter, and that matters because, in each problem, only a few of them really count.

The **test** is only used at the end, once, for the winner. If the choice were made on the test,
the test would become just more training: the winner would be the one that got luckiest on it.

## Hyperparameters

The "hyperparameters" of the search itself, on the Agents screen:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `budget` | How many candidates to measure, counting the screening and the tuning. Each one trains once per fold | 24 |
| `folds` | How many parts the training data is split into to measure each candidate | 5 |
| `finalists` | How many models from the screening move on to the tuning | 3 |
| `max_rows` | With more training rows than this, cross-validation uses a sample (the winner trains on all of them at the end) | 2000 |

And the request: the **target**, the **metric to optimise** (accuracy, balanced accuracy, F1, ROC
AUC or log-loss in classification; R², RMSE or MAE in regression) and the **models** that take
part. Every model of the task joins to begin with, the teaching implementations included; untick
the ones to leave out, and each family's agent only tries the ticked models of its family. A model
that can't be measured with the chosen metric stays out, with the reason (ROC AUC and log-loss
need probabilities, and the perceptron gives none). With imbalanced classes, the data agent
suggests balanced accuracy: plain accuracy can be won by a model that always says the most
frequent class.

## When to use

- To quickly answer "which model works best for this?", with an honest, explained starting point.
- To compare model families on the same data, with the same preparation and the same folds: the
  comparison is fair.
- As a first step: the winner opens in Train & evaluate and in Explain, and any candidate can be
  trained by hand, to see it from the inside.
- The manual mode is still where each choice is learned. The agents only work with a target
  (classification and regression): in clustering there are no right predictions to compare.

## Pitfalls

- **The winner is a bit optimistic.** Among many similar candidates, the one that wins the
  cross-validation also had some luck with the folds. The test score usually comes out a little
  lower, and the more candidates, the more so.
- **Differences within the noise.** If the runner-up is less than one standard deviation behind
  the leader, in practice they're tied. In those cases, the simpler model is as good a choice as
  the winner (the baseline, like logistic regression, is on the board for that).
- **A short budget.** With few candidates, the tuning barely happens: the screening spends it all.
  With many, the search takes longer and the winner's optimism grows.
- **Little data.** With few rows, the folds are small and the deviation large: the choice changes
  with the seed. Try two seeds and compare.
- **There's only one preparation.** Every candidate uses the same preparation (that's what makes
  the comparison fair). If the data needs something else, prepare it by hand and ask the agents to
  use your preparation.

## What to watch in the search

On the **Agents** screen:

- the **conversation**: what the data agent decided and why, what each model agent tried and what
  the coordinator chose. It's written in words, not just numbers: each request says why (the
  reference, a first impression, how far behind the leader), and each reply says which settings
  changed (with the default next to the new value), what the score means ("it gets about 78 in
  every 100 rows right"), whether the change helped and whether the model memorises the training
  data;
- the **search** chart: each candidate in the order it was measured, with the spread between
  folds, and the line of the best so far. The line rises in steps: in the screening, when a better
  family shows up; in the tuning, when a draw hits the mark;
- **Each agent's best**: which family did best with this data, compared with the baseline;
- the **board**: the training mean next to the validation mean shows how much each candidate
  memorises; a candidate that failed stays there, with its reason;
- at the end, the winner's **test**, compared with what the cross-validation predicted.

## The AI agent (phase 13)

The agents above are rules: with the same seed, they always decide the same. With Ollama open, an
**AI agent** can join them (ADR 0042), taking part in the tuning alongside the family agents.

**The intuition.** Random search doesn't learn: the tenth draw doesn't know what the previous nine
measured. The AI agent does. On its turn, it gets the finalists, each one's search space and the
results of every candidate (the mean, the deviation and the training score), **thinks** and
proposes the next one: one of the finalists and the values to change, with the reason. For
example: "the tree has 100% on the training data and 78% on the validation, it's memorising; I'll
lower the depth". It's the idea behind Bayesian optimisation (using what's already been measured
to choose the next point), done by a language model instead of a statistical model.

**How it's put together** (`core/automl/ai_agent.py`):

1. `briefing` writes what the search knows so far, as text, with the numbers as in JSON;
2. `answer_schema` gives the answer's format (the model, the values and the reason). Ollama forces
   the model to answer in that format (structured output);
3. the model answers in chunks, with its reasoning apart (`think`). The reasoning goes into the
   conversation as it arrives: that's what you see growing on the screen;
4. `read_answer` validates the proposal: the model must be a finalist, the values must be within
   its search space and the combination can't have been measured already.

**What keeps it honest.** The coordinator does the measuring, with cross-validation on the training
data only, as for the others: the AI agent never sees the test or picks the winner. A proposal that
doesn't fit is turned down, with the reason, and spends no budget; after three refusals in a row,
it leaves the tuning.

**Pitfalls.**

- **A language model isn't deterministic**, even with the temperature at 0 and the same seed: on
  another computer it may answer something else. That's why the replies are kept in the result's
  request (`ai_replies`), and repeating the search with that request gives the same result,
  without Ollama.
- **Thinking isn't measuring.** The reasoning may sound right and be wrong; only the
  cross-validation says whether the proposal is good. Read the thought as a hypothesis, and the
  result as its test.
- **A search with AI is slower**: each proposal waits for the model (seconds, on a graphics card).

**What to watch.** The dashed "AI agent" node in the flowchart lights up while it thinks; in the
conversation, the "Thought" opens to read the whole reasoning; on the board, its proposals carry the
**AI** mark; and the closing reading compares its best proposal with the tuning's best draw.

## The analyst (phase 13)

When the search is over, you can ask about it: "Why did the winner win?", "Does the winner memorise
the training data?". The **analyst** answers. It's an AI agent too, but a different kind (ADR 0053).

**The intuition.** The AI agent gets everything at once, as text, and answers in a fixed format.
The analyst starts with nothing but your question: it doesn't see the search, it has **tools**, and
it chooses which ones to use. That's how most AI agents work today, with a loop:

1. it **thinks** about what it needs;
2. it **calls a tool**: the **board** (every candidate, best first, and each model's best),
   **one candidate** in detail, **two candidates** side by side, or the **data plan**;
3. it **reads** what the tool returned, which goes back into the conversation;
4. it thinks again, and calls another tool or **answers**.

**How it's put together** (`core/automl/analyst.py` and `analyst_tools.py`). Each tool is a
function with a name, a description and its arguments in JSON Schema: that's all the model sees of
it. When the model asks for one, the analyst runs it and sends the result back as a `tool` message.
After six rounds of tools, it has to answer with what it has.

**What keeps it honest.**

- **The tools only read.** They read what the search measured (the result and the agents'
  conversation): the analyst trains nothing, measures nothing new and sees nothing of the test
  except the winner's score, which the result already shows.
- **The tools do the maths.** A small language model gets arithmetic wrong and misreads a table of
  numbers. So the gap between two candidates, whether it's a tie in practice (smaller than the
  spread between folds), how many folds each one won and whether a model memorises the training
  data all come already worked out, written as the screen shows them.
- **The answer is checked.** Every number in it must be in what the tools returned. If one isn't,
  a warning says which: a language model can invent a number that sounds right, and here that
  shows.

**Pitfalls.**

- **Right numbers, wrong sentence.** The check only looks at numbers: "#10 had a higher training
  score, 80.8% against 85.8%" passes, because both numbers are real. Read the answer next to the
  tools it used: every step is there.
- **General knowledge sneaks in.** "Random forests reduce overfitting" may be true in general and
  say nothing about this search. The instructions ask for explanations from what was measured, but
  a model doesn't always follow them.
- **It's slow**: each round waits for the model. On a laptop graphics card, a question takes from
  half a minute to a minute.

**Why the board is one line per candidate.** The board has every candidate, not just the top ten:
to explain why the winner won, the analyst needs the whole search. Written as one object per
candidate, a board of 24 is long, and a small model mixes up the rows, giving one candidate
another's score. So each candidate is a single line, with its number, model and score together,
and a short list says each model once, with its best candidate.

**What to watch.** On the Agents screen, after a search, **Ask the analyst** is a conversation:
your questions on the right, the analyst on the left, and only the conversation scrolls. **Expand**
opens it full screen, like a chat. While it works, each step arrives as it comes: the reasoning
growing live, each tool with its arguments and **What it returned** (the JSON the model read). When
it answers, the steps fold into one line with the tools it used; opened, they list what it did (each
tool, then the answer), with the **Thought** behind each one folded underneath. Under the answer
there's **Copy** and, when every number passed the check, a **Numbers checked** badge; a number
that didn't pass gets a warning, in full. Its replies are kept with the answer: asking again with
them gives the same answer, without Ollama.
