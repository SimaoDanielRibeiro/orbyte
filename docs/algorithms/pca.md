# PCA (principal component analysis)

## Intuition

Data with many features doesn't fit on a chart. PCA looks for the **directions in which the data
varies most** and rewrites each point in those directions. The first component is the direction
of greatest variation; the second, the one of greatest variation among those perpendicular to the
first; and so on. Keeping the first two gives a 2D chart that holds as much as possible of the
data's "shape".

It's what the Prepare screen already uses for the 2D projection, here as a model, with the number
of components up to you.

## Essential maths

1. Centre the data: $X_c = X - \bar{x}$.
2. Compute the covariance matrix: $\Sigma = \frac{1}{n-1} X_c^\top X_c$.
3. The **eigenvectors** of $\Sigma$ are the principal components; the **eigenvalues**
   $\lambda_j$ are the variance of the data along each one.
4. Project: the new coordinates are $Z = X_c W_k$, with the $k$ components of largest
   $\lambda_j$ in the columns of $W_k$.

The **explained variance** of each component is $\lambda_j / \sum_l \lambda_l$: the share of the
total variation it keeps. The teaching version (`pca_numpy`) does exactly these sums with
`np.linalg.eigh`; scikit-learn's uses the singular value decomposition (SVD) of the centred data,
which gives the same components with more numerical stability. The sign of each component is
arbitrary; both pick the one that makes the component's largest value positive.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_components` | How many directions to keep. With 2, the data fits on a chart | 2 |
| `whiten` | In scikit-learn, divide each component by its standard deviation (they all end up with variance 1) | no |

## When to use

- To **see** data with many features on a 2D chart.
- To **simplify** before another model: fewer features, less noise, faster training.
- To understand which features go together (each component's weights).

## Pitfalls

- **Feature scale:** without standardisation, the feature with the largest numbers dominates the
  first component just because of its unit. Standardise in the preparation.
- **It's linear:** curved structures (a roll, a ring) get squashed. t-SNE shows the neighbours
  better, but it's neither linear nor reversible.
- **Variance isn't importance:** the direction that varies most may not be the one that separates
  the classes.
- **The components are mixtures** of all the features: naming them isn't always easy.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen you see the **map**: the training points on
the first two components, coloured by the reference column, if you picked one on the Prepare
screen. In the evaluation, each component's explained variance and the "neighbour trust" say how
much of the original data the map kept.
