import os
import torch
import copy
import numpy as np
import h5py
from tqdm import tqdm

from tensordict import TensorDict
from torchmil.utils import build_adj, normalize_adj, add_self_loops


def default_read_file(file_path: str, key_type: str) -> np.ndarray:
    """
    Default function to read a file from disk. It supports .npy and .h5 files.

    Arguments:
        file_path: Path to the file to read.
        key_type: Type of the key to read. Must be in ['features', 'labels', 'inst_labels', 'coords'].

    Returns:
        data: Data read from the file.
    """
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == ".npy":
        return np.load(file_path)
    elif file_ext == ".h5":
        with h5py.File(file_path, "r") as f:
            return f[key_type][:]
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


class ProcessedMILDataset(torch.utils.data.Dataset):
    r"""
    This class represents a general MIL dataset where the bags have been processed and saved as numpy or .h5 files.
    It enforces strict data availability for core components, failing fast if expected files are missing.

    **MIL processing and directory structure.**
   The dataset expects pre-processed bags saved as individual numpy or .h5 files.
   
    - A feature file should yield an array of shape `(bag_size, ...)`, where `...` represents the shape of the features.
    - A label file should yield an array of shape arbitrary shape, e.g., `(1,)` for binary classification.
    - An instance label file should yield an array of shape `(bag_size, ...)`, where `...` represents the shape of the instance labels.
    - A coordinates file should yield an array of shape `(bag_size, coords_dim)`, where `coords_dim` is the dimension of the coordinates.
    
    **Bag keys and directory structure.**
    The dataset can be initialized with a list of bag keys, which are used to choose which data to load.
    This dataset expects the following directory structure:

    ```
    features_path/ (if "X" in bag_keys)
    ├── bag1.ext
    ├── bag2.ext
    └── ...
    labels_path/ (if "Y" in bag_keys)
    ├── bag1.ext
    ├── bag2.ext
    └── ...
    inst_labels_path/ (if "y_inst" in bag_keys)
    ├── bag1.ext
    ├── bag2.ext
    └── ...
    coords_path/ (if "coords" or "adj" in bag_keys)
    ├── bag1.ext
    ├── bag2.ext
    └── ...
    ```

    **Adjacency matrix.**
    If the coordinates of the instances are available, the adjacency matrix will be built using the Euclidean distance between the coordinates.
    Formally, the adjacency matrix $\mathbf{A} = \left[ A_{ij} \right]$ is defined as:

    $$A_{ij} = \begin{cases}
    d_{ij}, & \text{if } \left\| \mathbf{c}_i - \mathbf{c}_j \right\| \leq \text{dist_thr}, \\
    0, & \text{otherwise},
    \end{cases} \quad d_{ij} = \begin{cases}
    1, & \text{if } \text{adj_with_dist=False}, \\
    \exp\left( -\frac{\left\| \mathbf{x}_i - \mathbf{x}_j \right\|}{d} \right), & \text{if } \text{adj_with_dist=True}.
    \end{cases}$$

    where $\mathbf{c}_i$ and $\mathbf{c}_j$ are the coordinates of the instances $i$ and $j$, respectively, $\text{dist_thr}$ is a threshold distance,
    and $\mathbf{x}_i \in \mathbb{R}^d$ and $\mathbf{x}_j \in \mathbb{R}^d$ are the features of instances $i$ and $j$, respectively.
    
    **How bags are built.**
    When the `__getitem__` method is called, the bag is built as follows (pseudocode):

    1. The `__getitem__` method is called with an index.
    2. The bag name is retrieved from the list of bag names.
    3. The `_build_bag` method is called with the bag name:
        3.1. The `_build_bag` method loads the bag from disk using the `_load_bag` method. This method loads the features, labels, instance labels and coordinates from disk using the `_load_features`, `_load_labels`, `_load_inst_labels` and `_load_coords` methods.
        3.2. If the coordinates have been provided, it builds the adjacency matrix using the `_build_adj` method.
    4. The bag is returned as a dictionary containing the keys defined in `bag_keys` and their corresponding values.
    This behaviour can be extended or modified by overriding the corresponding methods.

    **Custom file reading.**
    By default, the dataset supports reading .npy and .h5 files using the `default_read_file` function. 
    However, users can provide a custom file reading function through the `read_file_fn` argument in the constructor. 
    This function must take as input the file path and the key type (one of 'features', 'labels', 'inst_labels', 'coords') and return the corresponding data as a numpy array. 
    ```python
        def custom_read_file(file_path: str, key_type: str) -> np.ndarray:
            # Custom logic to read the file and return the data as a numpy array
            ...    
    ```
    """

    def __init__(
        self,
        features_path: str = None,
        labels_path: str = None,
        inst_labels_path: str = None,
        coords_path: str = None,
        bag_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        file_ext: str = ".npy",
        dist_thr: float = 1.5,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = False,
        read_file_fn: callable = default_read_file,
        verbose: bool = True,
    ) -> None:
        """
        Class constructor.

        Arguments:
            features_path: Path to the directory containing the features.
            labels_path: Path to the directory containing the bag labels.
            inst_labels_path: Path to the directory containing the instance labels.
            coords_path: Path to the directory containing the coordinates.
            bag_keys: List of keys to load the bags data. The TensorDict returned by the `__getitem__` method will have these keys. Possible keys are:
                - "X": Load the features of the bag.
                - "Y": Load the label of the bag.
                - "y_inst": Load the instance labels of the bag.
                - "adj": Load the adjacency matrix of the bag. It requires the coordinates to be loaded.
                - "coords": Load the coordinates of the bag.
            file_ext: File type of files to be loaded. Can be '.npy' or '.h5'.
            bag_names: List of bag names to load. If None, all bags from the `features_path` are loaded.
            dist_thr: Distance threshold for building the adjacency matrix.
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the instance features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
            read_file_fn: Function to read the files from disk. It must take as input the file path and the key type (one of 'features', 'labels', 'inst_labels', 'coords') and return the corresponding data as a numpy array.
            verbose: If True, warning messages are displayed
        """
        super().__init__()

        self.features_path = features_path
        self.labels_path = labels_path
        self.inst_labels_path = inst_labels_path
        self.coords_path = coords_path
        self.file_ext = file_ext
        self.bag_names = bag_names
        self.bag_keys = bag_keys
        self.dist_thr = dist_thr
        self.adj_with_dist = adj_with_dist
        self.norm_adj = norm_adj
        self.load_at_init = load_at_init
        self.verbose = verbose
        self.read_file_fn = read_file_fn

        if "X" in self.bag_keys and self.features_path is None:
            raise ValueError("features_path must be provided if 'X' is in bag_keys")
        if "Y" in self.bag_keys and self.labels_path is None:
            raise ValueError("labels_path must be provided if 'Y' is in bag_keys")
        if "y_inst" in self.bag_keys and self.inst_labels_path is None:
            raise ValueError(
                "inst_labels_path must be provided if 'y_inst' is in bag_keys"
            )
        if "coords" in self.bag_keys and self.coords_path is None:
            raise ValueError("coords_path must be provided if 'coords' is in bag_keys")
        if "adj" in self.bag_keys and self.coords_path is None:
            raise ValueError("coords_path must be provided if 'adj' is in bag_keys")

        if self.bag_names is None:
            if self.features_path is None:
                raise ValueError("features_path must be provided if bag_names is None")

            self.bag_names = [
                file
                for file in os.listdir(self.features_path)
                if os.path.splitext(file)[1] == self.file_ext
            ]
            self.bag_names = [os.path.splitext(file)[0] for file in self.bag_names]

            if len(self.bag_names) == 0:
                raise ValueError("No bags found in features_path")

        self.bag_names = sorted(self.bag_names)

        self.loaded_bags = {}
        if self.load_at_init:
            pbar = tqdm(
                self.bag_names,
                desc="Loading bags",
                unit="bag",
                disable=not self.verbose,
            )
            for name in pbar:
                self.loaded_bags[name] = self._build_bag(name)

    def _read_file(self, file_path: str, key_type: str) -> np.ndarray:
        """
        Read a file from disk.

        Arguments:
            file_path: Path to the file to read.
            key_type: Type of the key to read. Must be in ['features', 'labels', 'inst_labels', 'coords'].

        Returns:
            data: Data read from the file.
        """
        try:
            data = self.read_file_fn(file_path, key_type)
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        return data

    def _load_features(self, name: str) -> np.ndarray:
        """
        Load the features of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            features: Features of the bag.
        """
        features_file = os.path.join(self.features_path, name + self.file_ext)
        features = self._read_file(features_file, "features")
        return features

    def _load_labels(self, name: str) -> np.ndarray:
        """
        Load the label of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            label: Label of the bag.
        """

        label_file = os.path.join(self.labels_path, name + self.file_ext)
        # label = np.load(label_file)
        label = self._read_file(label_file, "labels")
        return label

    def _load_inst_labels(self, name: str) -> np.ndarray:
        """
        Load the instance labels of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            inst_labels: Instance labels of the bag.
        """
        inst_labels_file = os.path.join(self.inst_labels_path, name + self.file_ext)
        inst_labels = self._read_file(inst_labels_file, "inst_labels")
        return inst_labels

    def _load_coords(self, name: str) -> np.ndarray:
        """
        Load the coordinates of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            coords: Coordinates of the bag.
        """
        coords_file = os.path.join(self.coords_path, name + self.file_ext)
        coords = self._read_file(coords_file, "coords")
        return coords

    def _load_bag(self, name: str) -> dict[str, torch.Tensor]:
        """
        Load a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            bag_dict: Dictionary containing the features ('X'), label ('Y'), instance labels ('y_inst') and coordinates ('coords') of the bag.
        """

        bag_dict = {}
        if "X" in self.bag_keys:
            bag_dict["X"] = self._load_features(name)

        if "Y" in self.bag_keys:
            bag_dict["Y"] = self._load_labels(name)

        if "y_inst" in self.bag_keys:
            bag_dict["y_inst"] = self._load_inst_labels(name)

        if "coords" in self.bag_keys or "adj" in self.bag_keys:
            bag_dict["coords"] = self._load_coords(name)

        return bag_dict

    def _build_bag(self, name: str) -> dict[str, torch.Tensor]:
        """
        Build a bag from the features, labels, instance labels and coordinates. First, it loads the bag from disk using `_load_bag`, then it builds the adjacency matrix using `_build_adj`.

        Arguments:
            name: Name of the bag to build.

        Returns:
            bag_dict: Dictionary containing the features ('X'), label ('Y'), instance labels ('y_inst') and coordinates ('coords') of the bag.
        """
        bag_dict = self._load_bag(name)

        if "adj" in self.bag_keys and bag_dict["coords"] is not None:
            edge_index, edge_weight, norm_edge_weight = self._build_adj(bag_dict)
            if self.norm_adj:
                edge_val = norm_edge_weight
            else:
                edge_val = edge_weight

            bag_dict["adj"] = torch.sparse_coo_tensor(
                edge_index,
                edge_val,
                (bag_dict["coords"].shape[0], bag_dict["coords"].shape[0]),
            ).coalesce()
            bag_dict["coords"] = torch.from_numpy(bag_dict["coords"])

        for key in ["X", "Y", "y_inst"]:
            if key in bag_dict:
                bag_dict[key] = torch.from_numpy(bag_dict[key])

        return bag_dict

    def _build_adj(
        self, bag_dict: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the adjacency matrix of a bag.

        Arguments:
            bag_dict: Dictionary containing the features ('X'), label ('Y'), instance labels ('y_inst') and coordinates ('coords') of the bag.

        Returns:
            edge_index: Edge index of the adjacency matrix, of shape `(2, n_edges)`.
            edge_weight: Edge weight of the adjacency matrix, of shape `(n_edges,)`.
            norm_edge_weight: Normalized edge weight of the adjacency matrix, of shape `(n_edges,)`.
        """

        bag_size = bag_dict["coords"].shape[0]
        if self.adj_with_dist:
            edge_index, edge_weight = build_adj(
                bag_dict["coords"], bag_dict["X"], dist_thr=self.dist_thr
            )
        else:
            edge_index, edge_weight = build_adj(
                bag_dict["coords"], None, dist_thr=self.dist_thr
            )
        norm_edge_weight = normalize_adj(edge_index, edge_weight, n_nodes=bag_size)
        if bag_size == 1:
            edge_index, norm_edge_weight = add_self_loops(
                edge_index, norm_edge_weight, bag_size
            )

        return edge_index, edge_weight, norm_edge_weight

    def __len__(self) -> int:
        """
        Returns:
            Number of bags in the dataset
        """
        return len(self.bag_names)

    def __getitem__(self, index: int) -> TensorDict:
        """
        Arguments:
            index: Index of the bag to retrieve.

        Returns:
            bag_dict: Dictionary containing the keys defined in `bag_keys` and their corresponding values.

                - X: Features of the bag, of shape `(bag_size, ...)`.
                - Y: Label of the bag.
                - y_inst: Instance labels of the bag, of shape `(bag_size, ...)`.
                - adj: Adjacency matrix of the bag. It is a sparse COO tensor of shape `(bag_size, bag_size)`. If `norm_adj=True`, the adjacency matrix is normalized.
                - coords: Coordinates of the bag, of shape `(bag_size, coords_dim)`.
        """

        bag_name = self.bag_names[index]

        if bag_name in self.loaded_bags.keys():
            bag_dict = self.loaded_bags[bag_name]
        else:
            bag_dict = self._build_bag(bag_name)
            self.loaded_bags[bag_name] = bag_dict

        return_bag_dict = {
            key: bag_dict[key] for key in self.bag_keys if key in bag_dict
        }

        return TensorDict(return_bag_dict)

    def get_bag_labels(self) -> list:
        """
        Returns:
            List of bag labels.
        """
        return [self._load_labels(name) for name in self.bag_names]

    def get_bag_names(self) -> list:
        """
        Returns:
            List of bag names.
        """
        return self.bag_names

    def subset(self, indices: list) -> "ProcessedMILDataset":
        """
        Create a subset of the dataset.

        Arguments:
            indices: List of indices to keep.

        Returns:
            subset_dataset: Subset of the dataset.
        """

        new_dataset = copy.deepcopy(self)
        new_dataset.bag_names = [self.bag_names[i] for i in indices]
        new_dataset.loaded_bags = {
            k: v
            for k, v in new_dataset.loaded_bags.items()
            if k in new_dataset.bag_names
        }

        return new_dataset
