import numpy as np

from .binary_classification_dataset import BinaryClassificationDataset

from ..utils.common import read_csv, keep_only_existing_files


class CAMELYON16MILDataset(BinaryClassificationDataset):
    r"""
    CAMELYON16 dataset for Multiple Instance Learning (MIL). Download it from [Hugging Face Datasets](https://huggingface.co/datasets/torchmil/Camelyon16_MIL/).

    **About the Original CAMELYON16 Dataset.**
    The original [CAMELYON16 dataset](https://camelyon17.grand-challenge.org/Data/) contains WSIs of hematoxylin and eosin (H&E) stained lymph node sections. The task is to identify whether each slide contains metastatic tissue and to localize it precisely. The dataset includes high-quality pixel-level annotations marking the metastases.

    **Dataset Description.**
    We have preprocessed the whole-slide images (WSIs) by extracting relevant patches and computing features for each patch using various feature extractors.

    - A **patch** is labeled as positive (`patch_label=1`) if more than 50% of its pixels are annotated as metastatic.
    - A **WSI** is labeled as positive (`label=1`) if it contains at least one positive patch.

    This means a slide is considered positive if there is any evidence of metastatic tissue.

    **Directory Structure.**
    After extracting the contents of the `.tar.gz` archives, the following directory structure is expected:

    ```
    root
    ├── patches_{patch_size}
    │ ├── features
    │ │ ├── features_{features_name}
    │ │ │ ├── wsi1.npy
    │ │ │ ├── wsi2.npy
    │ │ │ └── ...
    │ ├── labels
    │ │ ├── wsi1.npy
    │ │ ├── wsi2.npy
    │ │ └── ...
    │ ├── patch_labels
    │ │ ├── wsi1.npy
    │ │ ├── wsi2.npy
    │ │ └── ...
    │ ├── coords
    │ │ ├── wsi1.npy
    │ │ ├── wsi2.npy
    │ │ └── ...
    └── splits.csv
    ```
    Each `.npy` file corresponds to a single WSI. The `splits.csv` file defines train/test splits for standardized experimentation.
    """

    def __init__(
        self,
        root: str,
        features: str = "UNI",
        partition: str = "train",
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        patch_size: int = 512,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Arguments:
            root: Path to the root directory of the dataset.
            features: Type of features to use. Must be one of ['UNI', 'resnet50_bt'].
            partition: Partition of the dataset. Must be one of ['train', 'test'].
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'coords'].
            patch_size: Size of the patches. Currently, only 512 is supported.
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the patches features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
            verbose: If True, print additional information during loading.
        """
        features_path = f"{root}/patches_{patch_size}/features/features_{features}/"
        labels_path = f"{root}/patches_{patch_size}/labels/"
        patch_labels_path = f"{root}/patches_{patch_size}/patch_labels/"
        coords_path = f"{root}/patches_{patch_size}/coords/"

        self.patch_size = patch_size

        splits_file = f"{root}/splits.csv"
        dict_list = read_csv(splits_file)
        wsi_names = [row["bag_name"] for row in dict_list if row["split"] == partition]
        wsi_names = list(set(wsi_names))
        wsi_names = keep_only_existing_files(features_path, wsi_names)

        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            inst_labels_path=patch_labels_path,
            coords_path=coords_path,
            bag_names=wsi_names,
            bag_keys=bag_keys,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
            verbose=verbose,
        )

    def _load_coords(self, name):
        coords = super()._load_coords(name)
        if coords is not None:
            coords = coords / self.patch_size
            min_coords = np.min(coords, axis=0)
            coords = coords - min_coords
            coords = coords.astype(int)
        return coords

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        return BinaryClassificationDataset._load_bag(self, name)
