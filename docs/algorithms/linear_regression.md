# Linear regression

## Intuition

Linear regression predicts a **number** (the price of a house, an exam grade) as a weighted sum
of the features plus a constant: "each extra room adds so much to the price, each kilometre to
the centre takes off so much". With a single feature, the model is a straight line; with two, a
plane; with more, a hyperplane, the same idea in more dimensions.

Learning is choosing the weights that put the line as close as possible to the training points.
"Close" is measured with the **squared error**: the vertical distance from each point to the
line, squared.

## Essential maths

For an example with features $x$, the prediction is

$$\hat y = w \cdot x + b$$

and training looks for the $w$ and $b$ that minimise the **mean squared error** (MSE) on the
training data:

$$\text{MSE}(w, b) = \frac{1}{n} \sum_i (\hat y_i - y_i)^2$$

The square makes large errors count much more: missing by 10 costs 100 times more than missing
by 1. That's why a single point far from the trend can pull the line.

There are two ways to reach the best weights.

**In one go, with the least-squares formula** (`linear_regression`, from scikit-learn). The MSE
is a "bowl" (a quadratic function of the weights), and the bottom of the bowl has a formula:
$w = (X^\top X)^{-1} X^\top y$. scikit-learn computes it in a numerically more stable way,
without inverting the matrix.

**Bit by bit, by gradient descent** (`linear_regression_gd`, the lab's teaching version). It
starts with $w = 0$ and $b = 0$ and, in each **epoch** (one pass over all the training examples):

1. compute the residuals with the current weights: $r_i = \hat y_i - y_i$;
2. compute the gradient, the direction in which the error rises most:
   $\nabla_w = \frac{1}{n} X^\top r$ and $\nabla_b = \frac{1}{n} \sum_i r_i$ (it's the gradient
   of $\text{MSE}/2$, which has the same minimum and a cleaner formula);
3. take a step the opposite way: $w \leftarrow w - \eta \, \nabla_w$, where $\eta$ is the
   **learning rate**.

As the bowl has only one bottom, gradient descent always gets there, as long as the steps aren't
too large. The limit depends on how steep the bowl is in its steepest direction: with
standardised features, rates up to about 1 usually work; above that, each step jumps over the
bottom and lands higher on the other side, and the error **blows up**.

At the end, **R²** sums up the quality: $R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i -
\bar y)^2}$, the share of the target's variation that the model explains. Always predicting the
mean gives $R^2 = 0$; a model worse than that has a negative $R^2$.

## Hyperparameters

In both versions:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `fit_intercept` | Whether it learns the intercept $b$ (without it, the line has to go through the origin) | yes |

Only in gradient descent:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `learning_rate` | The size of each gradient descent step: too small drags on, too large blows up | 0.1 |
| `max_epochs` | The most passes over the training data | 1000 |
| `tol` | Stops when the gradient (measured on the target's scale) falls below this: it reached the bottom | 1e-6 |

## When to use

- As the **first model** for any regression: it's fast and gives the baseline that more complex
  models have to beat.
- When the relationship between the features and the target is roughly a straight line, or when
  the prediction has to be **explained**: each weight says how much the target changes per unit
  of the feature.

## Pitfalls

- **It's linear.** A curved relationship (a price that rises faster and faster with the area) is
  badly described: the residuals show a pattern instead of noise. Polynomial features help.
- **Outliers.** The squared error gives a single extreme point a huge weight: look at the
  outliers in the cleaning before training.
- **Feature scale.** Gradient descent suffers a lot without standardisation: the bowl becomes a
  long, narrow valley, where the rate that suits one feature is too large for another. The
  training blows up (the "loss blew up" warning) or takes thousands of epochs. Standardise in
  the preparation.
- **Redundant features.** If one feature is a combination of others (one-hot with every category
  plus the intercept, for example), there are infinitely many equally good solutions: the weights
  can no longer be read one by one and gradient descent moves very slowly along the bowl's "flat
  bottom". The predictions stay good; regularisation (Ridge) fixes it.
- **Extrapolating.** Outside the range of the training data, the line keeps going up (or down)
  without limit, even when reality doesn't.

## What to watch in training

On the **Train & evaluate** screen, the teaching version shows, epoch by epoch:

- the **loss curve** (the MSE on the training data): it falls fast at first and then more and
  more slowly, as it gets close to the bottom of the bowl;
- with a single feature, the **line turning and shifting** over the cloud of points until it
  fits it; with more features, the predicted value at each point of the 2D projection's plane.

Try the learning rate on the synthetic regression data: at 0.01 the training drags on and
doesn't reach the bottom in 1000 epochs; at 0.5 it gets there in a little over 20; at 3 it blows
up in the first epochs, and the training stops with the "loss blew up" warning. The Lab screen's
"Learning rate" demo draws the path of the weights on the bowl itself.

The scikit-learn version trains in one go: compare the weights of both in **What it learned**.
When gradient descent converges, they match.
