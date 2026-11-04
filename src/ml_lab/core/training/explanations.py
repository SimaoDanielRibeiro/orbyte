"""Explicações didáticas do treino: os textos do painel "Porquê?" do ecrã Treinar e avaliar.

Como as da camada de dados, vivem no backend, escritas uma só vez, e a interface pede-as
pela API.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

TRAINING_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "loss": Explanation(
            "Loss",
            "The loss (loss function) measures how wrong the model is on the training data, in "
            "a single number: the log-loss in classification, the mean squared error in "
            "regression. Training is searching for the parameters that make it as small as "
            "possible. A loss that falls fast and then levels off is a healthy training run; "
            "one that rises, or jumps from side to side, calls for a smaller learning rate.",
        ),
        "gradient_descent": Explanation(
            "Gradient descent",
            "The gradient says in which direction the loss rises most. Gradient descent takes "
            "steps the opposite way, the size of the **learning rate**: like walking down a "
            "hill in fog, always by the steepest side. Each pass over all the training examples "
            "is an **epoch**. Steps too small drag the training on; too large, they jump over "
            "the bottom and the loss blows up.",
        ),
        "snapshots": Explanation(
            "Snapshots",
            "A snapshot is a photo of the model halfway through training: the weights and the "
            "predictions on a grid that covers the data. Keeping every step would take too much "
            "memory, so at most 60 are kept, spread over the training (the first and the last "
            "always stay). Going through the snapshots in order is watching the model learn.",
        ),
        "decision_boundary": Explanation(
            "Decision boundary",
            "The decision boundary is the line where the model changes its mind: on one side "
            "it predicts one class, on the other the other (in logistic regression, where the "
            "probability is 50%). It's drawn on the plane of the 2D projection from the Prepare "
            "screen; the directions the plane doesn't show sit at the mean of the training "
            "data. With more than two features, it's therefore a slice of the true boundary, "
            "and some points can look as if they're on the wrong side when they aren't.",
        ),
        "embedding": Explanation(
            "The reduction map",
            "Each point is a training row in the new coordinates. If you picked a reference "
            "column on the Prepare screen, the colours are its classes: well-separated classes "
            "mean the reduction kept what tells them apart. The neighbour trust says whether "
            "what's together on the map was together in the data.",
        ),
        "neural_network": Explanation(
            "Neural network",
            "A dense neural network is a sequence of layers. Each neuron takes a weighted sum "
            "of its inputs, plus an intercept, and passes it through a non-linear "
            "**activation** (ReLU, tanh, sigmoid). Without the activation, several layers in a "
            "row would be just one linear sum, and the boundary would always be a straight "
            "line; with it, the network folds the space and draws curved boundaries. Training "
            "is adjusting all the weights at once so the loss goes down.",
        ),
        "backpropagation": Explanation(
            "Backpropagation",
            "To take a step, the network needs the gradient of the loss for each weight. "
            "**Backpropagation** computes them all at once, from the output back to the input, "
            "with the chain rule: one layer's gradient uses the next layer's. Training goes "
            "through the data in **batches**: for each batch, predictions, loss, gradients and "
            "one optimiser step (Adam adjusts each weight's step on its own).",
        ),
        "early_stopping": Explanation(
            "Early stopping",
            "Part of the training data is set aside, the **validation**, and the network never "
            "learns from it. While the validation loss falls, the network is learning what "
            "generalises; when it stops falling and the training loss keeps dropping, it starts "
            "memorising. Training stops after a few epochs without improvement (the "
            "**patience**) and the network keeps the weights from the best epoch.",
        ),
        "network_diagram": Explanation(
            "The network's layers",
            "Each circle is a neuron, from the input (on the left) to the output (on the "
            "right). In the hidden layers, the colour is the neuron's mean activation over the "
            "training examples. A **stuck** neuron (grey) always gives the same value, whatever "
            "the example, so it helps tell nothing apart. Change the step to watch the layers "
            "organise themselves during training.",
        ),
        "network_gradients": Explanation(
            "Each layer's gradients",
            "The size of the gradient of each layer's weights, epoch by epoch (on a "
            "logarithmic scale). Gradients that shrink a lot from layer to layer, down to the "
            "first, are the **vanishing gradient**: the first layers barely learn. It happens "
            "with the sigmoid in deep networks, and it's one of the reasons ReLU is the usual "
            "activation.",
        ),
        "network_stuck": Explanation(
            "Stuck neurons",
            "The share of each layer's neurons that always give the same value. In ReLU, a "
            "neuron whose sum stayed negative always gives zero and the gradient through it is "
            "zero: it died, and it doesn't come back. In tanh and the sigmoid, a saturated "
            "neuron (always near the extremes) barely learns. Many stuck neurons call for a "
            "smaller learning rate.",
        ),
        "network_activations": Explanation(
            "Activations",
            "The distribution of each layer's neuron outputs, at one training step. In ReLU, a "
            "tall bar at zero is the neurons switched off for those examples; in tanh and the "
            "sigmoid, tall bars at the extremes are saturated neurons.",
        ),
    }
)
