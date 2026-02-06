import numpy as np

from .processed_mil_dataset import ProcessedMILDataset


class VideoClassificationDataset(ProcessedMILDataset):
    r"""
    This class represents a dataset of videos for Multiple Instance Learning (MIL).

    **MIL and Video Classification.**
    Videos are sequences of frames that capture motion and temporal information.
    In the context of MIL, a video is considered a bag, and the frames are considered instances.

    **Directory structure.**
    It is assumed that the bags have been processed and saved as numpy files.
    For more information on the processing of the bags, refer to the [`ProcessedMILDataset` class](processed_mil_dataset.md).
    This dataset expects the following directory structure:

    ```
    features_path
    ├── video1.npy
    ├── video2.npy
    └── ...
    labels_path
    ├── video1.npy
    ├── video2.npy
    └── ...
    inst_labels_path
    ├── video1.npy
    ├── video2.npy
    └── ...
    ```

    **Order of the frames and the adjacency matrix.**
    This dataset assumes that the frames of the video frames are ordered.
    An adjacency matrix $\mathbf{A} = \left[ A_{ij} \right]$ is built using this information:

    \begin{equation}
    A_{ij} = \begin{cases}
    d_{ij}, & \text{if } \lvert i - j \rvert = 1, \\
    0, & \text{otherwise},
    \end{cases} \quad d_{ij} = \begin{cases}
    1, & \text{if } \text{adj_with_dist=False}, \\
    \exp\left( -\frac{\left\| \mathbf{x}_i - \mathbf{x}_j \right\|}{d} \right), & \text{if } \text{adj_with_dist=True}.
    \end{cases}
    \end{equation}

    where $\mathbf{x}_i \in \mathbb{R}^d$ and $\mathbf{x}_j \in \mathbb{R}^d$ are the features of instances $i$ and $j$, respectively.
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        frame_labels_path: str = None,
        video_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
    ) -> None:
        """
        Class constructor.

        Arguments:
            features_path: Path to the directory containing the matrices of the videos
            labels_path: Path to the directory containing the labels of the videos.
            frame_labels_path: Path to the directory containing the labels of the frames.
            video_names: List of the names of the videos to load. If None, all videos in the `features_path` directory are loaded.
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'coords'].
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the frames features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
        """

        dist_thr = 1.10
        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            inst_labels_path=frame_labels_path,
            coords_path="",
            bag_names=video_names,
            bag_keys=bag_keys,
            adj_with_dist=adj_with_dist,
            dist_thr=dist_thr,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
        )

    def _add_coords(self, bag_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Add coordinates to the bag dictionary.

        Arguments:
            bag_dict: Dictionary containing the features, label and instance labels of the bag.

        Returns:
            bag_dict: Dictionary containing the features, label, instance labels and coordinates of the bag.
        """
        bag_size = bag_dict["X"].shape[0]
        bag_dict["coords"] = np.arange(0, bag_size).reshape(-1, 1)

        return bag_dict

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        """
        Load a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            bag_dict: Dictionary containing the features, label, instance labels and coordinates of the bag.
        """
        bag_dict = {}
        if "X" in self.bag_keys:
            bag_dict["X"] = self._load_features(name)

        if "Y" in self.bag_keys:
            bag_dict["Y"] = self._load_labels(name)

        if "y_inst" in self.bag_keys:
            bag_dict["y_inst"] = self._load_inst_labels(name)

        if "coords" in self.bag_keys or "adj" in self.bag_keys:
            bag_dict = self._add_coords(bag_dict)
        return bag_dict
