# Ridge and Lasso

## Intuition

They're linear regression with the weights **held back**. With many features, few rows or
features that are very alike, linear regression gives huge weights, which cancel each other out
and change a lot from one sample to another. Regularisation adds a cost to large weights: the
model only uses them if it's worth it.

- **Ridge** shrinks all the weights, but none reaches zero.
- **Lasso** sets the weights of the least useful features to zero, so it also **picks** the
  features.

## Essential maths

Both minimise the squared error plus a penalty, with strength $\alpha$:

$$\text{Ridge (L2):}\quad \frac{1}{n}\lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_2^2 \qquad\qquad \text{Lasso (L1):}\quad \frac{1}{2n}\lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_1$$

(scikit-learn writes Ridge without the $\frac{1}{n}$, which only changes the scale of $\alpha$).
With $\alpha = 0$ they're linear regression again; with a huge $\alpha$, the weights go to zero
and the model predicts the mean.

Ridge has a closed form, $w = (X^\top X + \alpha I)^{-1} X^\top y$: adding $\alpha$ to the
diagonal makes the matrix always invertible, even with redundant features. Lasso doesn't.
scikit-learn uses coordinate descent, which optimises one weight at a time. The $\lVert w
\rVert_1$ has a "kink" at zero, and that's why Lasso's weights end up *exactly* at zero: the kink
holds them there.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `alpha` | The regularisation strength: larger = smaller weights (and, in Lasso, more weights at zero) | 1.0 (Ridge) / 0.1 (Lasso) |
| `max_iter` | In Lasso, the optimiser's maximum iterations | 1000 |
| `fit_intercept` | Whether it learns the intercept $b$ (the intercept is never penalised) | yes |

## When to use

- **Ridge:** correlated features (a house's several size measurements), or almost as many
  features as rows. It's the safe choice when you want to use every feature.
- **Lasso:** when you suspect that only a few features matter and want to know which.

## Pitfalls

- **Feature scale:** the penalty treats all the weights the same, but the weight of a feature in
  the thousands is naturally small. Standardise in the preparation.
- **Lasso with correlated features** picks one almost at random and sets the others to zero:
  don't read "weight zero" as "has nothing to do with it".
- **The right `alpha` depends on the data:** pick it with cross-validation. The Lab screen shows
  the test R² rising and falling again as alpha grows.
- **Convergence warning in Lasso:** raise `max_iter` or standardise the features.

## What to watch in training

Both train in one go. In **What it learned**, compare the weights with linear regression's: in
Ridge they all shrink; in Lasso some disappear. The Lab screen's "Regularisation" demo shows each
weight's path as alpha grows.
