# One-Class SVM

## Intuition

The One-Class SVM learns from (almost entirely) normal data only and draws the **tightest boundary
around it**. When predicting, whatever falls outside the boundary is anomalous. It's the SVM idea
(a boundary with a margin, drawn by the support vectors), with a single class: separating the data
from the origin, in the kernel's space.

## Essential maths

The One-Class SVM looks for the $w$ and $\rho$ that minimise

$$\frac{1}{2}\lVert w \rVert^2 + \frac{1}{\nu n} \sum_i \xi_i - \rho \qquad \text{with } w \cdot \phi(x_i) \ge \rho - \xi_i,\ \xi_i \ge 0$$

where $\phi$ is the kernel's mapping (with RBF, a "hill" around each support vector). The decision
function $w \cdot \phi(x) - \rho$ is positive inside the boundary. The parameter $\nu$ is, at the
same time, an upper limit on the share of training points outside the boundary and a lower limit
on the share of support vectors.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `nu` | The most training points outside the boundary: in practice, the expected share of anomalies | 0.05 |
| `kernel` | The shape of the boundary around the normal data: `rbf`, `linear` or `poly` | `rbf` |
| `gamma` | In the RBF kernel, each point's reach: large, the boundary hugs each point; small, it's smooth | 0.5 |

## When to use

- When the normal data forms one or a few compact areas and you want a smooth boundary around
  them.
- With medium-sized datasets (like any SVM, it gets slow with many rows).

## Pitfalls

- **Feature scale:** the RBF kernel uses distances. Always standardise.
- **A large `gamma`** draws an island around each training point: any new point looks anomalous.
- **Anomalies in the training data** pull the boundary outwards. `nu` gives them some slack.
- **Many rows:** training grows faster than the number of rows.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen, the colours show the anomaly score on the
plane: you see the boundary hugging the clouds of points. Try `gamma`: small gives a smooth shape;
large, islands around each point.
