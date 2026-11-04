# Hierarchical clustering (agglomerative)

## Intuition

It starts with each point alone in its own group and, step by step, **joins the two closest
groups**, until only one is left. The history of the joins forms a tree, the **dendrogram**:
cutting it at a certain height gives a certain number of groups. Instead of one split, it gives a
whole hierarchy: species within genera, genera within families.

## Essential maths

The algorithm is always the same; what changes is how the distance between two groups $A$ and $B$
is measured (the **linkage**):

- **single**: the smallest distance between a point of $A$ and one of $B$ (it follows chains of
  points);
- **complete**: the largest distance (it prefers compact groups);
- **average**: the mean of the distances between every pair;
- **Ward**: the increase in the variance within the groups if $A$ and $B$ were joined (it gives
  round, balanced groups, like k-means).

At each step, it joins the pair with the smallest distance. The height of each join in the
dendrogram is that distance. To get $k$ groups, the last $k - 1$ joins are undone.

The lab keeps the dendrogram in the model's state. For new points (which scikit-learn doesn't
predict), it uses the group of the closest training point.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_clusters` | How many groups the dendrogram is cut into | 3 |
| `linkage` | The distance between groups: `ward`, `complete`, `average` or `single` | `ward` |

## When to use

- When the **structure on several levels** matters, and not just one split.
- With **little data** (up to a few thousand rows), to see the dendrogram and choose the number
  of groups by looking at the heights of the joins.

## Pitfalls

- **Memory:** it keeps the distance between every pair of points ($n^2/2$). The lab refuses more
  than 5,000 training rows.
- **Irreversible decisions:** a bad join early on is never undone.
- **Single linkage chains:** two groups connected by a line of points join too early.
- **Feature scale:** the distances mix all the features. Standardise.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen you see the groups on the plane and, in
the evaluation, the **dendrogram**: the last 30 joins, with the cut line that gives the chosen
groups. A join much higher than the others is a natural place to cut.
