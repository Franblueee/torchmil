from typing import Any

import torch
from tensordict import TensorDict


def init_criterion(n_outputs: int, criterion: torch.nn.Module) -> torch.nn.Module:
    if n_outputs > 1 and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        return torch.nn.CrossEntropyLoss()
    return criterion


def compute_criterion_loss(
    criterion: torch.nn.Module,
    Y_pred: torch.Tensor,
    Y: torch.Tensor,
    n_outputs: int,
) -> torch.Tensor:
    if n_outputs == 1:
        return criterion(Y_pred.float(), Y.float().view_as(Y_pred))
    return criterion(Y_pred.float(), Y.long().view(-1))


def squeeze_binary_logits(Y_pred: torch.Tensor, n_outputs: int) -> torch.Tensor:
    if n_outputs == 1:
        return Y_pred.squeeze(-1)
    return Y_pred


def get_args_names(fn):
    args_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    # remove self from arg_names if exists
    if "self" in args_names:
        args_names = args_names[1:]
    return args_names


class MILModel(torch.nn.Module):
    r"""
    Base class for Multiple Instance Learning (MIL) models in torchmil.

    Subclasses should implement the following methods:

    - `forward`: Forward pass of the model. Accepts bag features (and optionally other arguments) and returns the bag label prediction (and optionally other outputs).
    - `compute_loss`: Compute inner losses of the model. Accepts bag features (and optionally other arguments) and returns the output of the forward method a dictionary of pairs (loss_name, loss_value). By default, the model has no inner losses, so this dictionary is empty.
    - `predict`: Predict bag and (optionally) instance labels. Accepts bag features (and optionally other arguments) and returns label predictions (and optionally instance label predictions).
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the module.
        """
        super(MILModel, self).__init__()

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
        """
        raise NotImplementedError

    def compute_loss(
        self, Y: torch.Tensor, X: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, dict]:
        """
        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss values.
        """

        out = self.forward(X, *args, **kwargs)
        return out, {}

    def predict(
        self, X: torch.Tensor, return_inst_pred: bool = False, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        raise NotImplementedError


class MILModelWrapper(MILModel):
    """
    A wrapper class for MIL models in torchmil.
    It allows to use all models that inherit from `MILModel` using a common interface:

    ```python
    model_A = ... # forward accepts arguments 'X', 'adj'
    model_B = ... # forward accepts arguments 'X''
    model_A_w = MILModelWrapper(model_A)
    model_B_w = MILModelWrapper(model_B)

    bag = TensorDict({'X': ..., 'adj': ..., ...})
    Y_pred_A = model_A_w(bag) # calls model_A.forward(X=bag['X'], adj=bag['adj'])
    Y_pred_B = model_B_w(bag) # calls model_B.forward(X=bag['X'])
    ```
    """

    def __init__(self, model: MILModel) -> None:
        super().__init__()
        self.model = model

    """
    Arguments:
        model: MILModel instance to wrap.
    """

    def forward(self, bag: TensorDict, **kwargs) -> Any:
        """
        Arguments:
            bag: Dictionary containing one key for each argument accepted by the model's `forward` method.

        Returns:
            out: Output of the model's `forward` method.
        """
        arg_names = get_args_names(self.model.forward)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model(**arg_dict, **kwargs)

    def compute_loss(self, bag: TensorDict, **kwargs) -> tuple[Any, dict]:
        """
        Arguments:
            bag: Dictionary containing one key for each argument accepted by the model's `forward` method.

        Returns:
            out: Output of the model's `compute_loss` method.
        """
        arg_names = get_args_names(self.model.compute_loss)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model.compute_loss(**arg_dict, **kwargs)

    def predict(self, bag: TensorDict, **kwargs) -> Any:
        """
        Arguments:
            bag: Dictionary containing one key for each argument accepted by the model's `forward` method.

        Returns:
            out: Output of the model's `predict` method.
        """
        arg_names = get_args_names(self.model.predict)
        arg_dict = {k: bag[k] for k in bag.keys() if k in arg_names}
        return self.model.predict(**arg_dict, **kwargs)
