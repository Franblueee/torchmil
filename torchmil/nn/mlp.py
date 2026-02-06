import torch


def get_activation(name):
    """
    Get torch activation function by name.
    """
    if "relu" in name:
        return torch.nn.ReLU()
    elif "sigmoid" in name:
        return torch.nn.Sigmoid()
    elif "tanh" in name:
        return torch.nn.Tanh()
    else:
        return None


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) class.
    """

    def __init__(
        self, input_size=512, linear_sizes=[100, 50], activations=["relu", "relu"]
    ) -> None:
        """
        Arguments:
            input_size (int): Size of the input features.
            linear_sizes (list of int): List containing the sizes of each linear layer.
            activations (list of str): List containing the activation functions for each layer.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.linear_sizes = linear_sizes
        self.activations = activations
        layers = [torch.nn.Linear(self.input_size, linear_sizes[0])]
        if activations[0] not in ["None", "none"]:
            layers.append(get_activation(activations[0]))

        for i in range(1, len(linear_sizes)):
            layers.append(torch.nn.Linear(linear_sizes[i - 1], linear_sizes[i]))
            if activations[i] not in ["None", "none"]:
                layers.append(get_activation(activations[i]))

        # Filter out None values that might have been added
        layers = [layer for layer in layers if layer is not None]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        """
        Forward pass through the MLP.

        Arguments:
            X (torch.Tensor): Output tensor of shape `(batch_size, ...)`

        Returns:

            torch.Tensor: Output of the MLP with shape `(batch_size, linear_sizes[-1])`
        """
        return self.net(X)
