import pytest
import numpy as np
import pandas as pd
from torchmil.datasets import TADMILDataset

@pytest.fixture
def mock_tad_dataset(tmp_path):
    root = tmp_path
    features = "resnet50"
    features_path = root / f"features/features_{features}"
    labels_path = root / "labels"
    frame_labels_path = root / "frame_labels"

    # Create directories
    for path in [features_path, labels_path, frame_labels_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Create a dummy video
    video_name = "video1"
    num_frames = 5
    feature_dim = 128
    
    # Save dummy .npy files
    # Features: (5 frames, 128 dimensions)
    np.save(features_path / f"{video_name}.npy", np.random.rand(num_frames, feature_dim))
    # Label: Scalar (0 or 1)
    np.save(labels_path / f"{video_name}.npy", np.array(1))
    # Frame Labels: (5 frames,)
    np.save(frame_labels_path / f"{video_name}.npy", np.random.randint(0, 2, size=(num_frames,)))

    # Create a splits.csv
    splits = pd.DataFrame({"bag_name": [video_name], "split": ["train"]})
    splits.to_csv(root / "splits.csv", index=False)

    return {
        "root": str(root),
        "features": features,
        "partition": "train",
        "bag_keys": ["X", "Y", "y_inst", "coords"], # Explicitly requesting y_inst for test
        "adj_with_dist": False,
        "norm_adj": True,
        "load_at_init": False,
    }

def test_tad_init(mock_tad_dataset):
    dataset = TADMILDataset(**mock_tad_dataset)
    assert hasattr(dataset, "_load_bag")
    # Verify the path construction logic inside init
    assert dataset.features_path.endswith("features/features_resnet50/")
    assert dataset.labels_path.endswith("labels/")
    assert dataset.inst_labels_path.endswith("frame_labels/")

def test_tad_load_bag(mock_tad_dataset):
    dataset = TADMILDataset(**mock_tad_dataset)
    bag = dataset._load_bag("video1")

    assert isinstance(bag, dict)
    assert "X" in bag
    assert "Y" in bag
    assert "y_inst" in bag # Should be present because we added it to bag_keys in fixture
    assert "coords" in bag # Added automatically by _add_coords
    
    # Verify shapes match the created data
    assert bag["X"].shape[0] == 5 # 5 frames
    assert bag["coords"].shape[0] == 5
    assert bag["y_inst"].shape[0] == 5

def test_tad_split_filtering(mock_tad_dataset, tmp_path):
    """Test that it respects the partition (train vs test) defined in splits.csv"""
    
    # Add a 'test' video to the existing structure
    root = tmp_path
    features_path = root / "features/features_resnet50"
    
    video_test = "video_test"
    np.save(features_path / f"{video_test}.npy", np.random.rand(5, 128))
    
    # Update splits.csv to include a test video
    splits = pd.DataFrame({
        "bag_name": ["video1", video_test], 
        "split": ["train", "test"]
    })
    splits.to_csv(root / "splits.csv", index=False)

    # Initialize with partition='test'
    params = mock_tad_dataset.copy()
    params["partition"] = "test"
    
    dataset = TADMILDataset(**params)
    
    # Should only contain 'video_test'
    assert len(dataset.bag_names) == 1
    assert dataset.bag_names[0] == "video_test"