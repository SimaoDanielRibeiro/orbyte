# t-SNE

## Intuition

t-SNE (*t-distributed stochastic neighbor embedding*) draws a **2D map** where similar points stay
together. It doesn't look for directions, like PCA: it looks for positions on the plane that
respect who's a neighbour of whom. It's great for seeing groups in data with many features. But
the map misleads in everything other than "these are close": the distances between groups and the
size of each group mean nothing.

## Essential maths

1. For each pair of points in the original data, the probability that they're neighbours,
   $p_{ij}$: a Gaussian around each point, with its width chosen so that the "effective number of
   neighbours" is the **perplexity**.
2. On the map, the same idea with a Student's t distribution (with long tails), $q_{ij}$.
3. Look for positions on the map that minimise the Kullback-Leibler divergence,
   $\mathrm{KL}(P \,\|\, Q) = \sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}}$, by gradient descent.

The long tails of the t distribution push different groups far from each other on the map. That's
why the groups look so well separated, and also why the distances between them can't be trusted.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `perplexity` | Roughly how many neighbours each point tries to keep close. Small: many tiny groups; large: the global structure, more smudged. It must be smaller than the number of rows | 30 |
| `max_iter` | The most gradient steps to lay out the map | 1000 |

## When to use

- To **explore visually** data with many features and see whether there are groups.
- After a PCA down to a few dozen components, on very large data.

## Pitfalls

- **It doesn't work for new points:** it doesn't learn a function, it only places on the map the
  points it trained on. That's why the evaluation only has the training report.
- **Don't read distances or sizes:** only the neighbourhood is reliable.
- **Slow with a lot of data** (tens of thousands of rows take minutes).
- **The perplexity changes the whole map:** try several values before drawing conclusions.
- **Different seeds give different maps** (rotated, mirrored, with the groups in other places).

## What to watch in training

It trains in one go (scikit-learn doesn't show the iterations). On the **Train & evaluate** screen
you see the map, coloured by the reference column, if you picked one. Compare it with the PCA map
of the same data: in t-SNE the groups are much more separated, and the evaluation shows whether
the neighbours on the map were neighbours in the data (the "neighbour trust").
