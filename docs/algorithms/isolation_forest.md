# Isolation Forest

## Intuition

To isolate a point, you make random cuts: pick a feature and any value between its minimum and
its maximum, and split. Repeat on each side until the point is alone. An **anomalous** point, far
from the others, is isolated with **few cuts**. A normal point, in the middle of a cloud, needs
many. The Isolation Forest does this with many trees and uses the average number of cuts.

It's the opposite idea from most detectors: instead of describing the normal points, it looks for
the ones that are easy to set apart.

## Essential maths

Each isolation tree is built with a sample of the rows (256, by default), with random cuts. The
**depth** $h(x)$ at which the point $x$ ends up alone, averaged over the trees, gives the score:

$$s(x) = 2^{-\overline{h(x)} / c(n)}$$

where $c(n)$ is the expected average depth in a tree of $n$ points (to normalise). Close to 1 is
an anomaly; well below 0.5, a normal point. The threshold is chosen to flag the expected share of
anomalies (`contamination`).

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_estimators` | How many isolation trees are averaged | 100 |
| `contamination` | What share of the training data is expected to be anomalous: it sets the threshold so it flags that share | 0.05 |

## When to use

- As the **first anomaly detector**: fast, works with many features and a lot of data, and needs
  no standardisation (the cuts are on one feature at a time).

## Pitfalls

- **`contamination` decides how many get flagged:** if the true share is different, the threshold
  sits in the wrong place. The score is the same; only the cut changes.
- **Anomalies in a group** (several anomalies together) are no longer so easy to isolate.
- **Anomalies that only show in a combination of features** (a normal value for each feature, but
  a rare combination) are hard with cuts parallel to the axes.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen, the colours show the anomaly score at
each point of the plane: the areas far from the data look more "anomalous". The evaluation shows
the distribution of the scores with the threshold and, if you picked the reference column (for
example, in the "Anomalies" generator), the precision, the recall and the ROC AUC.
