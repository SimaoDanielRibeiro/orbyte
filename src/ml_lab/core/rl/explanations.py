"""Textos do painel "Porquê?" do ecrã Reinforcement learning (só se escrevem aqui)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

RL_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "reinforcement_learning": Explanation(
            "Reinforcement learning",
            "There's no dataset: an **agent** learns to act in an **environment**. At each step "
            "it sees the state it's in, picks an action and gets a **reward**. Nobody tells it "
            "which action was right; only at the end does it find out what paid off. Over many "
            "**episodes** (attempts, from start to finish), it learns which actions lead to "
            "more reward: it's learning by trial and error.",
        ),
        "rl_state": Explanation(
            "The state",
            "The agent sees where it is: the cell of the grid. It doesn't see the map or know "
            "where the goal is; it only knows what it has already learned about each cell, its "
            "Q values.",
        ),
        "rl_action": Explanation(
            "The action",
            "It picks where to go. With probability ε, it picks at random: it **explores**. The "
            "rest of the time, it picks the action with the highest Q: it **exploits** what it "
            "already knows. ε starts at 1 and falls during training.",
        ),
        "rl_reward": Explanation(
            "The reward",
            "The environment answers with the cell where the agent ended up and a reward: −0.01 "
            "for each step, −1 in a trap and +1 at the goal. The last two end the episode.",
        ),
        "rl_update": Explanation(
            "The update",
            "The agent compares what it expected, Q(s, a), with what it saw: the reward plus "
            "the best Q of the next cell, discounted by γ. The difference is the **surprise**, "
            "and Q moves a fraction α of it. On the grid, the colour of the cell changes: "
            "that's how the value of the goal spreads backwards.",
        ),
        "rl_update_network": Explanation(
            "The update",
            "The surprise is the same as in Q-learning: what the agent saw (the reward plus the "
            "best Q of the next cell, discounted by γ) minus what it expected, Q(s, a). But "
            "there's no table to correct. The step goes into the **replay memory**, and the "
            "network takes a training step on a batch of steps drawn from it at random. So the "
            "value of a cell changes a little at every step, even of cells the agent isn't on, "
            "and the colours on the grid shift more slowly and more smoothly than with a table.",
        ),
        "q_values": Explanation(
            "Q values and policy",
            "**Q(state, action)** is the reward the agent expects to earn until the end of the "
            "episode if it takes that action there, and then plays the best it knows. The "
            "**policy** is the rule for choosing: in each state, the action with the highest "
            "Q. At the start, every Q is zero and the agent knows nothing; the goal is to get "
            "them right.",
        ),
        "temporal_difference": Explanation(
            "The temporal-difference error",
            "After each step, Q-learning compares what it expected, Q(s, a), with what it saw: "
            "the reward plus the best Q of the next state, discounted. The difference is the "
            "**surprise**, and Q moves a fraction of it (the learning rate α). That's how the "
            "+1 of the goal spreads backwards, cell by cell, all the way to the start.",
        ),
        "exploration": Explanation(
            "Explore or exploit (ε-greedy)",
            "If the agent always picked the action with the highest Q, it would never find "
            "paths better than the first one it found. With **ε-greedy**, it picks at random "
            "with probability ε (exploring) and the best it knows the rest of the time "
            "(exploiting). ε starts high, when nothing is known, and falls during training.",
        ),
        "discount": Explanation(
            "The discount (γ)",
            "A future reward is worth less than one now: each step into the future multiplies "
            "it by γ. With γ = 0.95, reaching the goal 10 steps from now is worth 0.95¹⁰ ≈ 0.6. "
            "That's why the short path is worth more than the long one, and why a low γ makes "
            "a short-sighted agent that only cares about what's close.",
        ),
        "deep_q_network": Explanation(
            "DQN: a network in place of the table",
            "DQN learns the same Q values, but with a neural network, which allows too many "
            "states for a table. Training a network on the surprise of each step is unstable, "
            "so there are two extra pieces: the **replay memory**, from which batches of steps "
            "are drawn at random (consecutive steps are too alike), and the **target "
            "network**, a copy that's only updated now and then and that computes the targets "
            "(without it, the target would move at every training step).",
        ),
        "policy_map": Explanation(
            "How to read the grid",
            "The colour of each cell is its value: the highest Q of its actions, that is, how "
            "much the agent expects to earn from there. The arrows are the policy: the action "
            "it picks in each cell. The line is the path it would take from the start, without "
            "slipping. Step through the episodes to watch the value spread out from the goal.",
        ),
        "reward_curve": Explanation(
            "The reward per episode",
            "Each point is the reward of a training episode; the line is the moving average. "
            "The episodes are noisy because the agent still explores at random. The curve rises "
            "when the agent learns to reach the goal, and to do it in fewer steps.",
        ),
        "success_curve": Explanation(
            "The episodes that reach the goal",
            "The share of the latest episodes (the same window as the moving average) in which "
            "the agent reached the goal. It rises when the agent learns the path; while "
            "training, it rarely reaches 100%, because the agent keeps exploring at random now "
            "and then.",
        ),
        "q_table": Explanation(
            "How to read the Q table",
            "One grid per action: the colour of each cell is the Q of that action there, what "
            "the agent expects to earn if it takes it and then plays the best it knows. In "
            "each cell, the policy picks the action with the highest Q across the four grids.",
        ),
        "rl_evaluation": Explanation(
            "Training and test for an agent",
            "In **training**, the agent still explores: the metrics come from the latest "
            "episodes, with ε. On the **test**, it plays new episodes with the learned policy, "
            "without exploring or learning: it's what it knows how to do. The difference "
            "between the two is the price of exploring. On the cliff, it falls into the traps "
            "while training, but it gets across on the test.",
        ),
    }
)
"""Os temas do ecrã Reinforcement learning, pelo nome."""
