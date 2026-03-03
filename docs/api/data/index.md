# torchmil.data

**torchmil.data** provides a collection of utilities for handling data in Multiple Instance Learning (MIL) tasks.

## Bags in **torchmil**

!!! note
    The data representation in **torchmil** is inspired by the data representation in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs).

    See [this notebook](../../examples/data_representation.ipynb) for a detailed explanation of the data representation in **torchmil**.

In Multiple Instance Learning (MIL), a bag is a collection of instances.
In **torchmil**, a bag is represented as a [TensorDict](https://pytorch.org/tensordict/stable/index.html).
In most cases (e.g., the datasets provided in **torchmil**), a bag  will contain at least two keys:

- `bag['X']`: a tensor of shape `(bag_size, ...)` containing the instances in the bag. Usually, this tensor is called _bag feature matrix_, since these instances are feature vectors extracted from the raw representation of the instances, and therefore it has shape `(bag_size, feature_dim)`.
- `bag['Y']`: a tensor containing the label of the bag. In the simplest case, this tensor is a scalar, but it can be a tensor of any shape (e.g., in multi-class MIL).

Additionally, a bag may contain other keys. The most common ones in **torchmil** are:

- `bag['y_inst']`: a tensor of shape `(bag_size, ...)` containing the labels of the instances in the bag. In the pure MIL setting, this tensor is only used for evaluation purposes since the label of the instances are not known. However, some methods may require some sort of supervision at the instance level.
- `bag['adj']`: a tensor of shape `(bag_size, bag_size)` containing the adjacency matrix of the bag. This matrix is used to represent the relationships between the instances in the bag. The methods implemented in [**torchmil.models**](../models/index.md) allow this matrix to be a sparse tensor.
- `bag['coords']`: a tensor of shape `(bag_size, coords_dim)` containing the coordinates of the instances in the bag. This tensor is used to represent the absolute position of the instances in the bag.

The data representation in **torchmil** is designed to be flexible and to allow the user to add any additional information to the bags. The user can define new keys in the bags and use them in the models implemented in **torchmil**.

## More information

<!-- DATA_AUTOLIST -->
