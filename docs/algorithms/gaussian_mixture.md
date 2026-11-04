# Gaussian mixture

## Intuition

It assumes the data comes from k bell-shaped "clouds" (Gaussians), each with its own centre, its
own shape (round, elongated, tilted) and its own weight (what share of the points it produces).
The model discovers those clouds. Unlike k-means, a point doesn't belong to just one group: it
belongs to each cloud with a **probability**. A point between two clouds is half of each.

## Essential maths

The data's density is a weighted sum of Gaussians:

$$p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

The parameters are found with the **EM** algorithm (*expectation-maximization*), which repeats:

1. **E**: each cloud's responsibility for each point,
   $r_{ik} = \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) / p(x_i)$;
2. **M**: the means, covariances and weights that best explain those responsibilities (means
   weighted by $r_{ik}$).

No iteration ever lowers the mean **log-likelihood**, $\frac{1}{n}\sum_i \log p(x_i)$. The
algorithm stops when it barely rises any more. Like k-means, it finds a local maximum, which
depends on the starting point (scikit-learn starts from the result of a k-means).

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_components` | How many clouds (groups) | 3 |
| `covariance_type` | The shape of the clouds: `full` (each its own ellipse, tilted freely), `tied` (the same ellipse for all), `diag` (ellipses aligned with the axes) or `spherical` (circles) | `full` |
| `max_iter` | The most EM iterations | 100 |
| `tol` | Stops when the mean log-likelihood rises by less than this | $10^{-3}$ |

## When to use

- When the groups are **ellipses** of different sizes and tilts.
- When the **uncertainty** matters: the probability of each point belonging to each group.

## Pitfalls

- **Many parameters:** with `full`, each cloud has a complete covariance matrix ($d(d+1)/2$
  numbers). With many features and little data, use `diag` or `spherical`.
- **Clouds that shrink to a point** (the variance goes to zero and the likelihood to infinity):
  scikit-learn adds a small regularisation to the diagonal to prevent it.
- **Choosing k:** compare the silhouette, or criteria like the BIC.
- **The groups don't have to be Gaussian:** a moon isn't an ellipse.

## What to watch in training

On the **Train & evaluate** screen, each step is one EM iteration. The **ellipses** show each
cloud (at two standard deviations from the centre), and the crosses their centres. You see the
ellipses shift, stretch and turn until they fit the data. The loss is minus the mean
log-likelihood: it falls until it levels off.
