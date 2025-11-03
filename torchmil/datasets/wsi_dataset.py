import numpy as np

from .processed_mil_dataset import ProcessedMILDataset


class WSIDataset(ProcessedMILDataset):
    r"""
    This class represents a dataset of Whole Slide Images (WSI) for Multiple Instance Learning (MIL).

    **MIL and WSIs.**
    Whole Slide Images (WSIs) are high-resolution images of tissue samples used in digital pathology.
    Due to their large size, WSIs are usually divided into smaller patches.
    In the context of MIL, a WSI is considered a bag, and the patches are considered instances.

    **Patch and feature extraction.**
    Different tools are available to obtain the previous directory structure from a set of WSIs.
    For example, given a set of WSIs in the original format (e.g., .tif extension), patches can be extracted using tools such as [CLAM](https://github.com/mahmoodlab/CLAM).
    This tool outputs the coordinates of the patches, which can be used to extract the patches from the WSIs.
    Then, a feature vector can be extracted from each patch using a pretrained model.

    **Binary MIL for WSIs.**
    In binary classification, the label $Y$ of the WSI usually represents the presence ($Y=1$) or absence ($Y=0$) of a certain characteristic in the WSI.
    This characteristic can be present in one or more patches of the WSI, but the exact location of the characteristic is unknown.
    This translates into a patch having an unknown label $y_n$, which is positive ($y_n=1$) if it contains the characteristic, and negative ($y_n=0$) otherwise.
    Consequently, $Y = \max\left\{ y_1, y_2, \ldots, y_N \right\}$, where $N$ is the number of patches in the WSI.
    This means that the WSI is positive (contains the characteristic) if at least one of its patches is positive (contains the characteristic).
    In the case that the WSI has been annotated at the patch level, the instance labels $y_n$ can be used solely for evaluation purposes.
    See [`torchmil.datasets.BinaryClassificationDataset`](./binary_classification_dataset.md) for more information.

    **Directory structure.**
    It is assumed that the bags have been processed and saved as numpy or .h5 files.
    For more information on the processing of the bags, refer to the [`ProcessedMILDataset` class](processed_mil_dataset.md).
    This dataset expects the following directory structure:

    ```
    features_path
    ├── wsi1.ext
    ├── wsi2.ext
    └── ...
    labels_path
    ├── wsi1.ext
    ├── wsi2.ext
    └── ...
    inst_labels_path
    ├── wsi1.ext
    ├── wsi2.ext
    └── ...
    coords_path
    ├── wsi1.ext
    ├── wsi2.ext
    └── ...
    ```

    **Adjacency matrix.**
    If the coordinates of the patches are available, an adjacency matrix representing the spatial relationships between the patches is built.

    \begin{equation}
    A_{ij} = \begin{cases}
    d_{ij}, & \text{if } \left\| \mathbf{c}_i - \mathbf{c}_j \right\| \leq \text{dist_thr}, \\
    0, & \text{otherwise},
    \end{cases} \quad d_{ij} = \begin{cases}
    1, & \text{if } \text{adj_with_dist=False}, \\
    \exp\left( -\frac{\left\| \mathbf{x}_i - \mathbf{x}_j \right\|}{d} \right), & \text{if } \text{adj_with_dist=True}.
    \end{cases}
    \end{equation}

    where $\mathbf{c}_i$ and $\mathbf{c}_j$ are the coordinates of the patches $i$ and $j$, respectively, $\text{dist_thr}$ is a threshold distance,
    and $\mathbf{x}_i \in \mathbb{R}^d$ and $\mathbf{x}_j \in \mathbb{R}^d$ are the features of patches $i$ and $j$, respectively.
    If no `dist_thr` is provided, it is set to $\sqrt{2} \times \text{patch_size}$.
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        patch_labels_path: str = None,
        coords_path: str = None,
        wsi_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        file_type: str = ".npy",
        patch_size: int = 512,
        dist_thr: float = None,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
    ) -> None:
        """
        Class constructor.

        Arguments:
            features_path: Path to the directory containing the feature matrices of the WSIs.
            labels_path: Path to the directory containing the labels of the WSIs.
            patch_labels_path: Path to the directory containing the labels of the patches.
            coords_path: Path to the directory containing the coordinates of the patches.
            wsi_names: List of the names of the WSIs to load. If None, all the WSIs in the `features_path` directory are loaded.
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'coords'].
            file_type: File extension of the files containing the bags. Can be '.npy' or '.h5'.
            patch_size: Size of the patches.
            dist_thr: Distance threshold for building the adjacency matrix. If None, it is set to `sqrt(2) * patch_size`.
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the patches features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
        """
        if dist_thr is None:
            # dist_thr = np.sqrt(2.0) * patch_size
            dist_thr = np.sqrt(2.0)
        self.patch_size = patch_size
        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            inst_labels_path=patch_labels_path,
            coords_path=coords_path,
            bag_names=wsi_names,
            bag_keys=bag_keys,
            file_type=file_type,
            dist_thr=dist_thr,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
        )

    def _load_coords(self, name):
        coords = super()._load_coords(name)
        if coords is not None:
            coords = coords / self.patch_size
            min_coords = np.min(coords, axis=0)
            coords = coords - min_coords
            coords = coords.astype(int)
        return coords
