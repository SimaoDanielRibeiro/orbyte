# K-nearest neighbours (KNN)

## Intuition

"Tell me who your friends are, and I'll tell you who you are." KNN learns no weights and no
rules: it keeps the training examples and, to classify a new point, looks for the k examples
most similar to it and gives the class that shows up most often among them.

It's the classic example of an **instance-based** model: training is instant (it just
stores), and all the work is left for the prediction.

## Essential maths

For a point $x$:

1. compute the distance to each training example. The **Euclidean** distance is the straight
   line, $d(x, z) = \sqrt{\sum_j (x_j - z_j)^2}$; the **Manhattan** distance is the sum of the
   differences, $d(x, z) = \sum_j \lvert x_j - z_j \rvert$, like walking around city blocks;
2. keep the $k$ examples with the smallest distance;
3. count the votes for each class. With uniform weights, each neighbour is worth 1; with
   distance weights, it's worth $1/d$ (a neighbour at distance zero takes the whole vote).

The "probability" of each class is its share of the votes. The lab's teaching version
(`knn_numpy`) does these three steps in NumPy, in blocks of rows, so it doesn't fill the
memory.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_neighbors` | How many neighbours vote (the $k$). Few: a jagged boundary that follows the noise; many: a smooth boundary that can wipe out details | 5 |
| `weights` | `uniform` (every vote counts the same) or `distance` (the closest count more) | `uniform` |
| `metric` | The distance: `euclidean` (straight line) or `manhattan` (sum of the differences) | `euclidean` |

## When to use

- As a simple, honest baseline, especially with few features and many examples.
- When the boundary between the classes is irregular and you don't know what shape it has.

## Pitfalls

- **Feature scale:** the distance mixes all the features. A feature in the thousands (income)
  crushes one in units (age). Always standardise in the preparation.
- **A small k memorises the training data:** with $k = 1$, each training point gets itself
  right (100% on the training data) and creates an "island" around it (overfitting). The Lab
  screen shows the effect of k.
- **Many features:** in many dimensions, all points end up at roughly the same distance from
  each other (the "curse of dimensionality"), and "the closest" stops meaning "the most
  similar".
- **Slow predictions:** each prediction compares the point with every training example.
- **Ties:** with an even k, two classes can tie. The one with the lower number wins.

## What to watch in training

KNN trains in one go (it only stores the data). On the **Train & evaluate** screen you see the
final boundary: with a small k, islands appear around isolated points; with a large k, the
boundary is smooth. Compare `knn` with `knn_numpy` on the same data: the predictions are the
same.
