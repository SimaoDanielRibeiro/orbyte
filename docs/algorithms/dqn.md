# DQN (Deep Q-Network)

## Intuition

DQN learns the same thing as **Q-learning**, the **Q(s, a)** values, but keeps them in a **neural
network** instead of a table. The network takes the state and returns one Q per action. On a small
grid, a table would do; the network is what makes room for problems with too many states for a
table, like a game's images. That's how, in 2015, a single algorithm learned to play dozens of
Atari games from the pixels.

Training a network on the surprise of each step is unstable: the network chases a target that it
computes itself, and forgets what it learned from old steps. DQN adds two pieces to calm it down:

- a **replay memory**: each step goes into a memory, and the network learns from **batches drawn
  from it at random**. Consecutive steps are very alike (the same area of the grid); mixing them
  breaks that correlation and lets each step be reused several times;
- a **target network**: a frozen copy of the network that computes the targets. It's only updated
  now and then. Without it, each training step would also move the target, like chasing your own
  shadow.

## Essential maths

In Orbyte, the state goes into the network **one-hot**: a vector of zeros with a 1 at the state's
position. The network has two hidden layers with ReLU and one output per action:

$$\text{one-hot}(s) \rightarrow \text{ReLU} \rightarrow \text{ReLU} \rightarrow \big(Q(s, a_1), \dots, Q(s, a_4)\big)$$

At each step, after storing the step in the memory, a batch is drawn at random and, for each step
$(s, a, r, s')$ in the batch, the target is computed with the **target network** $\hat{Q}$:

$$y = r + \gamma \max_{a'} \hat{Q}(s', a') \qquad (y = r \text{ if the episode ended})$$

The loss is the **Huber** loss between $Q(s, a)$ and $y$: quadratic for small errors and linear for
large ones, so a huge surprise (an unexpected trap) doesn't give a huge gradient. The optimiser is
Adam. Every `target_update` training steps, the target network becomes a copy of the network.

Picking the actions works as in Q-learning: **ε-greedy**, with $\varepsilon$ falling from
`epsilon_start` to `epsilon_end`, over the `exploration_fraction` of the episodes.

## Hyperparameters

What it shares with Q-learning:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `episodes` | How many times the agent crosses the environment | 300 |
| `discount` | The discount $\gamma$: how much the future is worth | 0.95 |
| `epsilon_start` | The exploration in the first episode | 1 |
| `epsilon_end` | The exploration once it has come down | 0.05 |
| `exploration_fraction` | Over what part of the episodes $\varepsilon$ falls | 0.5 |

What's only in DQN:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `learning_rate` | Adam's step: too large, the network oscillates; too small, it takes long | 0.001 |
| `hidden_units` | Neurons in each of the two hidden layers | 64 |
| `batch_size` | How many steps from the memory go into each training step | 32 |
| `replay_size` | How many steps the memory keeps (the oldest leave first) | 5000 |
| `target_update` | Every how many training steps the target network is updated | 100 |

## When to use

- When there are too many states for a table: continuous positions, images, many variables.
- With few, discrete actions (for continuous actions, like a steering wheel's angle, there are
  other methods, like PPO).
- On a small grid, Q-learning is enough and much faster: DQN is here so you can see what changes
  when the table becomes a network.

## Pitfalls

- **Instability.** Without the replay memory and the target network, the training oscillates or
  diverges. Try `target_update` at 1 (no target network) and compare the reward per episode.
- **Slow.** Each step of the agent is also a training step for the network: DQN takes seconds
  where Q-learning takes tenths.
- **A small memory.** With `replay_size` close to `batch_size`, the batches go back to being
  consecutive, correlated steps, and the reason for having a memory is lost.
- **A high learning rate.** The network chases targets that move; too large a step makes the Q
  values blow up.
- **Randomness.** The starting weights, the exploration and the batches come from the seed:
  another seed gives another training run. Compare several seeds in History.

## What to watch in training

On the **Reinforcement learning** screen, as with Q-learning:

- the **reward per episode**, usually noisier than Q-learning's, because the network changes the Q
  of every state at once;
- the **grid**, with the value and the policy in each cell: the values spread out from the goal,
  but less tidily than in the table, because the network generalises (changing one weight moves
  several states);
- each episode's **loss**, the Huber loss: it falls when the Q values settle.

Before training one yourself, the **How an agent learns** lesson at the top of the screen shows a
DQN learning the cliff, step by step (choose it next to the title). Compare it with Q-learning's:
in the first episode the cells already have colours, because the network starts with random
weights, not zeros; and the value spreads more slowly and more smoothly, because each step only
moves the network a little, with a batch drawn from the replay memory.
