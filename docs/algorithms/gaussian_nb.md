# Gaussian naive Bayes

## Intuition

For each class, naive Bayes learns what its examples look like: the mean and the variance of
each feature. A *setosa* flower has short petals, with little variation; a *virginica* has long
petals. To classify a new flower, it asks: "with these measurements, which class is it most
likely to come from?"

It's **naive** because it assumes that, within each class, the features have nothing to do with
each other. That's almost never true, but it works surprisingly well.

## Essential maths

Bayes' rule gives the probability of each class $c$ given the point $x$:

$$p(c \mid x) \propto p(c)\, p(x \mid c)$$

$p(c)$ is the class's frequency in the training data. With the naive assumption, $p(x \mid c)$ is
the product of the probabilities of each feature, each with its own normal (Gaussian) curve:

$$p(x \mid c) = \prod_j \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\!\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

Training is just computing $\mu_{cj}$ and $\sigma_{cj}^2$, the mean and the variance of feature
$j$ in class $c$: one pass over the data, with no optimiser. The sums are done with logarithms,
so the products of small numbers don't turn into zero.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `var_smoothing` | A fraction of the largest variance added to all of them, so none is zero. A constant feature in a class would give probabilities of 0 or 1 | $10^{-9}$ |

## When to use

- With **little data**: it only learns means and variances, so it doesn't need many examples.
- As a lightning-fast baseline, and for text (with the multinomial variant, which counts words).

## Pitfalls

- **Correlated features count twice:** if two features say the same thing, the model treats them
  as two independent pieces of evidence and becomes overconfident. The probabilities tend to be
  extreme (almost 0 or almost 1), even when the predicted class is right.
- **It assumes normal curves:** a feature with two "humps" in the same class is badly described.
- **The boundary is quadratic:** curved, but always shaped like an ellipse or a parabola, never
  arbitrary.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen, the crosses are the means of each class
on the projection's plane, and the boundary runs between them, closer to the class with the
smaller variance. **What it learned** shows each class's means and variances.
