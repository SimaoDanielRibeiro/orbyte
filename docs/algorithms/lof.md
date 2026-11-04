# Local Outlier Factor (LOF)

## Intuition

LOF compares the **density around each point** with the density around its neighbours. A point in
an area much emptier than its neighbours' is anomalous. Because it's **local**, it finds anomalies
even when the groups have very different densities. A point a little away from a tight group is
suspicious; the same distance in a spread-out group is normal.

## Essential maths

For each point $x$, with its $k$ nearest neighbours $N_k(x)$:

1. the **local reachability density** is the inverse of the mean distance to the neighbours
   (using the "reachability distance", which smooths out very small distances):
   $\mathrm{lrd}(x) = 1 / \overline{d_{\mathrm{reach}}(x, N_k(x))}$;
2. the **LOF** is the ratio between the neighbours' density and the point's own:
   $$\mathrm{LOF}(x) = \frac{1}{k}\sum_{o \in N_k(x)} \frac{\mathrm{lrd}(o)}{\mathrm{lrd}(x)}$$

Close to 1, the point is as dense as its neighbours; well above 1, it's in an emptier area, and
it's anomalous. The lab uses scikit-learn's "novelty detection" mode (`novelty=True`), which can
score new points. The training points keep the score computed in training, because scoring them
again would count each point as its own neighbour.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_neighbors` | How many neighbours define the local density. Few: sensitive to noise; many: it misses anomalies in small groups | 20 |
| `contamination` | What share of the training data is expected to be anomalous: it sets the threshold so it flags that share | 0.05 |

## When to use

- When the data has **groups of different densities** and a global threshold (like the Isolation
  Forest's) can't suit them all.
- With medium-sized data and few features.

## Pitfalls

- **Feature scale:** it uses distances. Standardise.
- **`n_neighbors` changes everything:** it must be larger than the small groups of anomalies you
  want to find (otherwise, they're each other's neighbours and look dense).
- **Slow with a lot of data:** each score searches for a point's neighbours.
- **Many dimensions:** the distances, and the density with them, lose their meaning.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen, the colours show the anomaly score on the
plane. Unlike the Isolation Forest, the score rises right around the tight groups and more slowly
around the spread-out ones. In the evaluation, compare the distribution of the scores and the
threshold with those of the other detectors.
