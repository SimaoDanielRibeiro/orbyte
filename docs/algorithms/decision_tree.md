# Decision tree

## Intuition

A decision tree is a flowchart of yes-or-no questions about the features: "is the petal shorter
than 2.45 cm?" If yes, it's *setosa*; if not, it asks something else. Each path ends in a
**leaf**, which gives the answer: the most frequent class of the training examples that landed
there (classification) or their mean (regression).

It's one of the easiest models to explain: the path of a prediction reads like a rule.

## Essential maths

The tree is built from the top down. At each node, it tries every question "$x_j \le t$" (each
feature $j$, each threshold $t$ between two values) and picks the one that leaves the two sides
**purest**:

- in classification, with the **Gini** index, $G = 1 - \sum_k p_k^2$ (0 when there's only one
  class), or with the **entropy**, $H = -\sum_k p_k \log_2 p_k$;
- in regression, with the **squared error** around the mean of each side (or the absolute error
  around the median).

The choice is **greedy**: the best question now, without looking at the next ones. It repeats
on each side until the leaf is pure, until it reaches the maximum depth or until there are few
examples left.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `max_depth` | How many questions in a row (0 = no limit). Deep trees memorise the training data | 5 |
| `min_samples_leaf` | The minimum examples in each leaf: larger values avoid leaves made for a single point | 1 |
| `criterion` | The impurity measure: `gini` or `entropy` (classification); `squared_error` or `absolute_error` (regression) | `gini` / `squared_error` |

## When to use

- When the decision has to be **explained** as rules.
- With features of mixed types and scales: the questions are about one feature at a time, so
  the scale doesn't matter.
- As a building block for stronger models: random forests and boosting are made of trees.

## Pitfalls

- **Overfitting:** with no depth limit, the tree grows until each leaf has a single example
  and gets 100% of the training data right. Limit `max_depth` or raise `min_samples_leaf`.
- **Unstable:** changing a few training rows can change the first question and, with it, the
  whole tree. Forests fix this with the average of many trees.
- **Staircase boundaries:** the questions are always about one feature at a time, so the
  boundary is made of segments parallel to the axes. A diagonal boundary needs many steps.
- **In regression, it doesn't extrapolate:** outside the training data, it predicts the value
  of the outermost leaf, always the same.

## What to watch in training

On the **Train & evaluate** screen, each step is the tree with one more level of depth. You see
the boundary break into ever smaller rectangles, and the loss (the share of errors on the
training data, or the mean squared error) fall. Notice when the boundary starts drawing
rectangles around isolated points: that's overfitting arriving. **What it learned** shows the
tree drawn out, with each node's question.
