# Random forest

## Intuition

A single decision tree memorises the training data and changes a lot with small changes in the
data. But many **different** trees, each with its own errors, voting together, make far fewer
mistakes: the errors of some cancel out those of others. It's the "wisdom of crowds".

For the trees to be different, the random forest draws lots twice: each tree trains on a sample
of the rows, and each node can only choose among a few features picked at random.

## Essential maths

For each of the $B$ trees:

1. draw $n$ rows **with replacement** (the *bootstrap*): some appear several times and about a
   third ($e^{-1} \approx 37\%$) are left out;
2. build a deep tree; at each node, consider only a random subset of the features (for example,
   $\sqrt{p}$ of $p$ features).

The prediction is the majority vote (classification) or the mean (regression). Averaging models
with little correlation between them reduces the variance without raising the bias: that's
*bagging*.

**Out-of-bag error (OOB):** each row was left out of about a third of the trees. Those trees'
predictions for that row give an estimate of the error on new data, without spending a
validation set.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_estimators` | How many trees. More trees don't cause overfitting: they only make training slower, and past a certain point they stop helping | 100 |
| `max_depth` | The depth of each tree (0 = no limit). Deep trees are normal in a forest | 0 |
| `max_features` | How many features each node can try: `sqrt`, `log2` or `all`. Fewer features give more different trees | `sqrt` (classification) / `all` (regression) |

## When to use

- As the **first strong model** for tabular data: it works well with almost no tuning, doesn't
  need standardisation and measures the importance of the features.
- When there are interactions and non-linearities that a linear model misses.

## Pitfalls

- **Slower and heavier** than a single tree: it keeps every tree.
- **Hard to explain:** it's no longer a readable rule; use the feature importance and the
  partial dependence (Explain screen).
- **The "impurity" importance** (scikit-learn's) favours features with many values. The
  permutation importance, measured on the test, is more honest.
- **In regression, it doesn't extrapolate:** the average of trees stays stuck in the range of
  the training data.

## What to watch in training

On the **Train & evaluate** screen, each step adds more trees. The loss is the **OOB error**:
1 − OOB accuracy (classification) or 1 − OOB R² (regression). It falls fast with the first few
dozen trees and then levels off: that's why more trees do no harm, they just stop helping. The
boundary starts jagged (few trees) and gets smoother as the average settles.
