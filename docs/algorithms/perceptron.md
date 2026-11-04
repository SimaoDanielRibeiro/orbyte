# Perceptron

## Intuition

The perceptron (Rosenblatt, 1958) is the oldest linear classifier that learns on its own. It has
a line (or a plane) separating the classes and only moves it when it's wrong: if an example falls
on the wrong side, the line turns and shifts a little towards it. If it's right, nothing changes.

It's the direct ancestor of neural networks: an artificial neuron is a perceptron with a smoother
function in place of the "yes or no".

## Essential maths

With the classes coded as $y = +1$ and $y = -1$, the perceptron predicts the sign of
$w \cdot x + b$. In each epoch it goes through the examples and, for each one:

$$\text{if } y\,(w \cdot x + b) \le 0: \qquad w \leftarrow w + \eta\, y\, x \qquad b \leftarrow b + \eta\, y$$

where $\eta$ is the learning rate. The correction pushes the example's score the right way:
$y\,(w + \eta y x) \cdot x = y\, w \cdot x + \eta \lVert x \rVert^2$, larger than before.

**Convergence theorem:** if the classes can be separated by a straight line, the perceptron finds
a boundary with no errors after a finite number of corrections. If they can't, it never stops
correcting.

With more than two classes, one perceptron is trained per class ("this class against the
others") and the prediction is the class with the highest score $w_k \cdot x + b_k$.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `learning_rate` | The size of each correction ($\eta$). With the weights starting at zero, it only changes the scale of the weights and the intercept | 1.0 |
| `max_epochs` | The most passes over the training data | 100 |
| `shuffle` | Go through the examples in a different order each epoch (with the model's seed) | yes |
| `fit_intercept` | Whether it learns the intercept $b$ (without it, the boundary goes through the origin) | yes |

## When to use

- To **understand** how a classifier learns from its mistakes, and where neural networks came
  from.
- On data that a straight line can separate, where it finds a perfect boundary on the training
  data.

In practice, logistic regression or the linear SVM are better: they give probabilities or the
best margin, and handle overlapping classes well.

## Pitfalls

- **Classes that a straight line can't separate** (the moons, Iris with versicolor and
  virginica): the perceptron never stops correcting, and the final boundary depends on the epoch
  where it stopped. The model warns you ("it still made mistakes on the training data").
- **It gives no probabilities:** only the class. So there's no ROC curve and no log-loss.
- **Any boundary with no errors will do:** among several possible lines, it keeps the first one
  it found, often pressed against one of the classes. The SVM, by contrast, looks for the one
  furthest from both.
- **The order of the examples matters:** with `shuffle`, another seed gives another boundary.

## What to watch in training

On the **Train & evaluate** screen, each step is an epoch. The loss curve shows the share of
examples corrected in that epoch. On separable data it reaches zero and the training stops; on
the rest it oscillates without ever getting there. The boundary jumps from epoch to epoch,
because it only moves with the mistakes: that's the contrast with the smooth descent of logistic
regression's gradient descent.
