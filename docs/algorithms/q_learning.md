# Q-learning

## Intuition

There's no dataset and no right answers. There's an **agent** in an **environment**: at each step
it sees where it is (the **state**), picks an **action** and gets a **reward**. In Orbyte's grid
world, the agent earns +1 when it reaches the goal, loses 1 in a trap and pays 0.01 for each step.
Nobody tells it which action was the good one: it has to find out by playing many times (the
**episodes**).

Q-learning keeps a **table**: for each state and each action, a number **Q(s, a)**, the reward it
expects to earn from here to the end if it takes that action there and then plays the best it
knows. The **policy** is to pick, in each state, the action with the highest Q.

At the start the table is all zeros: the agent knows nothing and wanders at random. When it
reaches the goal for the first time, the action that took it there gains value. In the next
episode, the cell before that one gains too, because it leads to a cell that's already worth
something. The goal's +1 **spreads backwards**, cell by cell, all the way to the start. On the
Reinforcement learning screen's grid you can watch it happen, episode by episode.

## Essential maths

After each step (state $s$, action $a$, reward $r$, next state $s'$), Q-learning corrects a single
number in the table:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \big[\, r + \gamma \max_{a'} Q(s', a') - Q(s, a) \,\big]$$

- $r + \gamma \max_{a'} Q(s', a')$ is the **target**: the reward it saw, plus the best it expects
  from the next state. If the episode ended there (goal or trap), the target is just $r$: there's
  no future;
- the difference between the target and $Q(s, a)$ is the **temporal-difference error**: the
  surprise;
- $\alpha$ (`learning_rate`) is the share of the surprise that gets corrected;
- $\gamma$ (`discount`) is the **discount**: a reward $k$ steps away is worth $\gamma^k$. With
  $\gamma = 0.95$, the goal 10 steps away is worth $0.95^{10} \approx 0.6$, and that's why the
  short path is worth more than the long one.

To pick the action, the agent uses **ε-greedy**: with probability $\varepsilon$, a random action
(**exploring**); otherwise, the one with the highest Q (**exploiting**). $\varepsilon$ starts at
`epsilon_start` and falls in a straight line to `epsilon_end`, over the `exploration_fraction` of
the episodes. Without exploring, the agent would repeat the first path it found, even if there
were a better one.

The $\max$ in the target makes Q-learning an **off-policy** method: it learns the value of the
best policy, even while it behaves badly (exploring). That's the difference from SARSA, which uses
in the target the action it's actually going to take.

Running out of steps in an episode isn't a final state: the agent just stopped playing. So, in
that last step, the target still counts on the future.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `episodes` | How many times the agent crosses the environment: it's the training's limit | 500 |
| `learning_rate` | The $\alpha$: how much each step corrects Q. At 1, Q becomes only what was just seen; at 0.1, it changes slowly | 0.5 |
| `discount` | The $\gamma$: how much the future is worth. Close to 1, the agent thinks long term; close to 0, it's short-sighted | 0.95 |
| `epsilon_start` | The exploration in the first episode (1 = always at random) | 1 |
| `epsilon_end` | The exploration once it has come down: never zero, so it keeps trying things | 0.05 |
| `exploration_fraction` | Over what part of the episodes $\varepsilon$ falls from start to end | 0.5 |

## When to use

- When the states and actions can be counted and fit in a table: grids, small board games,
  inventory problems with few levels.
- To learn reinforcement learning: everything can be seen and explained, number by number.
- When there are too many states for a table (images, continuous positions), the table is replaced
  by a neural network: that's **DQN**.

## Pitfalls

- **Exploring too little.** With `epsilon_end` at zero and a short `exploration_fraction`, the
  agent sticks to the first path it found. In the maze, it may never find the way.
- **Environments with chance.** With slipping, the same step gives different rewards, and a high
  `learning_rate` makes Q jump with every surprise. With slipping, try 0.1.
- **The price of exploring.** On the cliff, Q-learning learns the path right along the traps, the
  shortest. During training, $\varepsilon$ still picks at random and the agent falls in often: the
  training reward stays below the test reward. That's what the report compares.
- **A low discount.** With $\gamma$ close to 0, the distant goal is worth almost nothing, and an
  agent far from it has nothing to guide it.
- **Too few episodes.** The value has to spread from the goal to the start; on a large map, that
  takes many episodes.

## What to watch in training

On the **Reinforcement learning** screen:

- the **reward per episode** starts low (the agent wanders around and pays for each step) and
  rises when it learns the path. The line is the moving average, because each episode is noisy;
- the **grid** colours each cell by its value and shows the policy (the arrows) and the path the
  agent would take. Step through the episodes and watch the value spread out from the goal;
- the **Q table**, action by action: next to the traps, the action that steps into one turns
  negative.

Try the cliff with slipping at 0 and at 0.1: with chance, the path right along the traps stops
being the best.
