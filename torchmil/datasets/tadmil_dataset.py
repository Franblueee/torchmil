import numpy as np

from .binary_classification_dataset import BinaryClassificationDataset
from .video_classification_dataset import VideoClassificationDataset

from ..utils.common import read_csv, keep_only_existing_files


class TADMILDataset(BinaryClassificationDataset, VideoClassificationDataset):
    r"""
    Traffic Anomaly Detection for Multiple Instance Learning (MIL).
    Download it from [Kaggle Datasets](https://www.kaggle.com/datasets/nikanvasei/traffic-anomaly-dataset-tad).


    **Dataset description.**
    We have preprocessed the Video by computing features for each frame using various feature extractors.

    - A **video** is labeled as positive (`label=1`) if it contains evidence of traffic anomaly.
    - A **video** is labeled as positive (`label=1`) if it contains at least one positive frame.

    This means a video is considered positive if there is any evidence of traffic anomaly.

    **Directory structure.**

    The following directory structure is expected:

    ```
    root
    ├── features
    │   ├── features_{features}
    │   │   ├── video1.npy
    │   │   ├── video2.npy
    │   │   └── ...
    ├── labels
    │   ├── video1.npy
    │   ├── video2.npy
    │   └── ...
    └── splits.csv
    ```

    Each `.npy` file corresponds to a video. The `splits.csv` file defines train/test splits for standardized experimentation.
    """

    def __init__(
        self,
        root: str,
        features: str = "resnet50",
        partition: str = "train",
        bag_keys: list = ["X", "Y", "adj", "coords"],
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
    ) -> None:
        """
        Arguments:
            root: Path to the root directory of the dataset.
            features: Type of features to use. Must be one of ['resnet18', 'resnet50', 'vit_b_32']
            partition: Partition of the dataset. Must be one of ['train', 'test'].
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'coords'].
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the patches features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
        """
        features_path = f"{root}/features/features_{features}/"
        labels_path = f"{root}/labels/"
        frame_labels_path = f"{root}/frame_labels/"

        splits_file = f"{root}/splits.csv"
        dict_list = read_csv(splits_file)
        video_names = [
            row["bag_name"] for row in dict_list if row["split"] == partition
        ]

        video_names = list(set(video_names))
        video_names = keep_only_existing_files(features_path, video_names)

        VideoClassificationDataset.__init__(
            self,
            features_path=features_path,
            labels_path=labels_path,
            frame_labels_path=frame_labels_path,
            bag_keys=bag_keys,
            video_names=video_names,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
        )

    def _load_bag(self, name: str) -> dict[str, np.ndarray]:
        bag_dict = BinaryClassificationDataset._load_bag(self, name)
        bag_dict = VideoClassificationDataset._add_coords(self, bag_dict)
        return bag_dict
