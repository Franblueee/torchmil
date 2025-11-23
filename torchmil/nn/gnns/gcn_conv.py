import torch

from torchmil.nn import LazyLinear


class GCNConv(torch.nn.Module):
    """
    Implementation of a Graph Convolutional Network (GCN) layer.

    Adapts the implementation from [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        add_self_loops: bool = False,
        learn_weights: bool = False,
        layer_norm: bool = False,
        normalize: bool = False,
        dropout: float = 0.0,
        activation: torch.nn.Module = torch.nn.Identity(),
        bias: bool = True,
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension.
            add_self_loops: Whether to add self-loops.
            learn_weights: Whether to use a linear layer after the convolution.
            layer_norm: Whether to use layer normalization.
            normalize: Whether to l2-normalize the output.
            dropout: Dropout rate.
            activation: Activation function to apply after the convolution.
            bias: Whether to use bias.
        """

        super(GCNConv, self).__init__()

        if out_dim is None or not learn_weights:
            out_dim = in_dim

        self.add_self_loops = add_self_loops
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout)
        self.normalize = normalize

        if learn_weights:
            self.fc = LazyLinear(in_dim, out_dim, bias=bias)
        else:
            self.fc = torch.nn.Identity()

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(out_dim)
        else:
            self.layer_norm = torch.nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Node features of shape (batch_size, n_nodes, in_dim).
            adj: Adjacency matrix of shape (batch_size, n_nodes, n_nodes).

        Returns:
            y: Output tensor of shape (batch_size, n_nodes, out_dim).
        """

        y = torch.bmm(adj, x)
        if self.add_self_loops:
            y += x
        y = self.fc(y)
        if self.normalize:
            y = torch.nn.functional.normalize(y, p=2, dim=-1)
        y = self.layer_norm(y)
        y = self.activation(y)
        y = self.dropout(y)

        return y
