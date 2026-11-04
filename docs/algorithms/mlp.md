# Dense neural network (MLP)

## Intuition

A logistic regression adds up the features with weights and decides which side of a **line**
each example is on. A neural network does that many times over, in a chain: each **neuron** of
the first layer is a small logistic regression, the second layer combines the first layer's
answers, and so on. Between layers there's a non-linear **activation**, which folds the space:
with it, the network draws **curved** boundaries, like the one that separates the two moons.

Nobody tells the network what each neuron should do. The weights start at random and the
training adjusts them all at once, step by step, so the loss goes down. The neurons end up
specialising: some react to one side of the plane, others to another, and the output combines
them.

The name **MLP** comes from *multilayer perceptron*: several layers of perceptrons, but with
smooth activations (which let gradients be computed), instead of the perceptron's step.

## Essential maths

A dense layer transforms the input vector $h$ (the features, or the previous layer's output)
like this:

$$h' = f(W h + b)$$

where $W$ is the weight matrix (one weight per input-neuron pair), $b$ the intercepts and $f$
the activation, applied to each neuron:

- **ReLU:** $f(z) = \max(0, z)$. It's the usual one: cheap, with a gradient of 1 whenever it's on;
- **tanh:** $f(z) = \tanh(z)$, between −1 and 1;
- **sigmoid:** $f(z) = 1 / (1 + e^{-z})$, between 0 and 1.

Without $f$, two layers in a row would be $W_2(W_1 h) = (W_2 W_1) h$: just another linear sum.
It's the activation that gives the network the power to approximate any shape.

The **output** depends on the task. In classification there's one neuron per class and the
**softmax** turns them into probabilities; the loss is the **cross-entropy**, the same log-loss
as logistic regression. In regression there's one neuron with no activation, and the loss is the
**mean squared error** (the target is standardised before training and the predictions go back
to the original scale).

To train, you need the gradient of the loss for each weight. **Backpropagation** computes them
all at once, from the output back to the input, with the chain rule: a layer's gradient comes
from the next layer's, multiplied by the derivatives of the activations. PyTorch does this sum
on its own (`backward`). Training goes through the data in **batches**; for each batch:

$$\text{predictions} \rightarrow \text{loss} \rightarrow \nabla \text{(backpropagation)} \rightarrow w \leftarrow w - \eta\, \nabla_w$$

**Adam** is a gradient descent that adjusts each weight's step by the history of its gradients:
it usually works first time, without tuning the rate. **SGD with momentum** follows the gradient
with some inertia.

The starting weights can't all be equal (the neurons would all learn the same thing), nor have
just any scale: too large saturates the activations, too small fades the signal from layer to
layer. So they're drawn at random with the activation's scale: **He** for ReLU and **Glorot** for
tanh and the sigmoid.

**Early stopping** sets part of the training data aside, the **validation**. While the
validation loss falls, the network learns what generalises; when it stops falling (and the
training loss keeps dropping), it starts memorising. Training stops after `patience` epochs
without improvement, and the network keeps the weights from the best epoch.

## Hyperparameters

The shape of the network:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `hidden_layers` | How many hidden layers: more layers, richer shapes (and harder to train) | 2 |
| `units` | Neurons in each hidden layer: few, a simple boundary; many, more risk of memorising | 16 |
| `activation` | The non-linearity: `relu`, `tanh` or `sigmoid` | `relu` |

The training:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `optimizer` | `adam` (adjusts each weight's step) or `sgd` (with momentum) | `adam` |
| `learning_rate` | The size of each step: too large, the loss jumps or blows up | 0.01 |
| `batch_size` | Examples per step: small batches give noisy steps, and many per epoch | 32 |
| `max_epochs` | The most passes over the training data | 300 |
| `alpha` | L2 regularisation: pulls the weights towards zero, so the boundary gets smoother | 0.0001 |
| `early_stopping` | Whether it stops when the validation stops improving | yes |
| `patience` | Epochs without improvement on the validation before it stops | 20 |
| `validation_fraction` | The part of the training data set aside for the validation | 0.1 |

## When to use

- When the boundary (or the relationship with the target) is **non-linear** and there's enough
  data: the moons, the circles, interactions between features that a linear model can't see.
- As a starting point for bigger networks (for images, for text): the training loop is the same.
- On small tabular data, forests and boosting usually win with less work; the network is worth it
  for what it teaches about training, gradients and activations.

## Pitfalls

- **Feature scale.** A network with features on very different scales trains badly: standardise
  in the preparation.
- **Learning rate.** Too large, the loss jumps or blows up (the training stops with the "loss blew
  up" warning); too small, it drags on. With Adam, 0.01 or 0.001 usually work.
- **Dead neurons.** A ReLU whose sum always stayed negative always gives zero, and the gradient
  through it is zero too: it doesn't come back. Many stuck neurons (in **The network's layers**)
  call for a smaller rate.
- **Vanishing gradient.** With the sigmoid (or tanh) in deep networks, the gradient shrinks from
  layer to layer and the first layers barely learn. You can see it in the gradients chart.
- **Overfitting.** With many neurons and little data, the network memorises the training data.
  Early stopping, `alpha` and smaller networks help; the validation loss rising while the training
  loss falls is the sign.
- **Randomness.** The starting weights and the order of the batches come from the seed: with
  another seed, the result changes a little. Compare runs with the same seed (or several seeds)
  in History.
- **Little data.** With few rows, the validation gets so few that early stopping stops at random
  (or is turned off, if none are left).

## What to watch in training

On the **Train & evaluate** screen, the network is photographed epoch by epoch:

- the **decision boundary** starts almost straight and curves around the moons;
- the **loss curve** shows the training and the validation: while they fall together, the network
  is learning; when the validation one levels off or rises, early stopping is near;
- the **network diagram** colours each neuron by its mean activation; the stuck ones appear faded.
  Change the step and watch the layers organise themselves;
- the **gradients chart** shows the size of each layer's gradient. Try 4 layers with the sigmoid
  and watch the first layers' gradient shrink;
- each layer's **activations**: in ReLU, the bar at zero is the neurons switched off; in tanh,
  tall bars at the extremes are saturated neurons.

Try it on the moons: with 2 neurons per layer, the boundary stays almost straight; with 16, it
curves around each moon; with a rate of 0.5, the neurons die and the loss stops falling.
