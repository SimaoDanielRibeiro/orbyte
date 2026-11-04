# K-means

## Intuition

K-means splits the data into k groups, each represented by its **centroid** (the group's middle
point). Each point belongs to the group of the closest centroid. The algorithm starts with k
centroids in chosen places and repeats two simple steps until they stop moving: assign each
point to the closest centroid, and move each centroid to the middle of its group.

## Essential maths

K-means looks for the centroids $\mu_1, \dots, \mu_k$ that minimise the **inertia**, the sum of
the squared distances from each point to the centroid of its group:

$$J = \sum_i \lVert x_i - \mu_{c(i)} \rVert^2$$

Lloyd's algorithm alternates:

1. **assign**: $c(i) = \arg\min_j \lVert x_i - \mu_j \rVert^2$;
2. **update**: $\mu_j = $ the mean of the points with $c(i) = j$.

No step ever raises $J$, so the algorithm always stops. But it stops at a **local minimum**,
which depends on where the centroids started. **k-means++** picks starting points far from each
other: the first at random, and each of the next with a probability proportional to the square
of the distance to the closest centroid already chosen.

The scikit-learn version repeats everything `n_init` times, from different starts, and keeps the
one with the lowest inertia. The teaching version (`kmeans_numpy`) runs just once, so you can see
the path.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_clusters` | How many groups (the $k$). K-means always finds k groups, even if the data doesn't have them | 3 |
| `init` | Where the centroids start: `k-means++` (spread out) or `random` (random points) | `k-means++` |
| `max_iter` | The most repetitions of assign and update | 300 |
| `tol` | Stops when the centroids barely move (a fraction of the data's variance) | $10^{-4}$ |
| `n_init` | In scikit-learn, how many different starts to try | 10 |

## When to use

- For **round groups of similar sizes**, like the blobs.
- As the first clustering to try: it's fast and easy to explain.

## Pitfalls

- **You have to choose k.** Compare the silhouette of several values, or look for the "elbow" in
  the inertia curve (which always falls with more groups).
- **Elongated or nested groups** (the moons, the circles) are split badly: k-means cuts
  everything into convex regions around the centroids. DBSCAN or the Gaussian mixture cope better.
- **Feature scale:** the distances mix all the features. Standardise.
- **Bad starts:** with `random`, two centroids can start in the same group. The teaching version
  keeps the result of that start (which is why scikit-learn's repeats several times).
- **Empty groups:** if a centroid is left with no points, the teaching version keeps it where it
  was and warns you. scikit-learn moves it far away.

## What to watch in training

In the teaching version, the **Train & evaluate** screen shows each iteration: the crosses (the
centroids) walk to the middle of their groups, and the coloured areas (the group each point of
the plane would get) change with them. The loss is the inertia, which only falls, and levels off
in a few iterations. With `random`, another seed may end in another split.
