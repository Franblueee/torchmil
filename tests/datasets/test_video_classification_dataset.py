import pytest
import numpy as np
from torchmil.datasets import VideoClassificationDataset

@pytest.fixture
def mock_video_data(tmp_path):
    features_path = tmp_path / "features"
    labels_path = tmp_path / "labels"
    frame_labels_path = tmp_path / "frame_labels"

    for p in [features_path, labels_path, frame_labels_path]:
        p.mkdir(parents=True, exist_ok=True)

    video_name = "vid1"
    num_frames = 5
    feature_dim = 64
    
    # Save dummy .npy files
    # Features: (5 frames, 64 features)
    np.save(features_path / f"{video_name}.npy", np.random.rand(num_frames, feature_dim))
    # Label: Scalar
    np.save(labels_path / f"{video_name}.npy", np.array(0))
    # Frame Labels: (5 frames,)
    np.save(frame_labels_path / f"{video_name}.npy", np.random.randint(0, 2, size=(num_frames,)))

    return {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "frame_labels_path": str(frame_labels_path),
        "video_names": [video_name],
        "bag_keys": ["X", "Y", "coords", "y_inst"],
        "adj_with_dist": False,
        "norm_adj": True,
        "load_at_init": False,
    }

def test_video_dataset_init(mock_video_data):
    dataset = VideoClassificationDataset(**mock_video_data)
    # Check that hardcoded defaults in __init__ are set correctly
    assert dataset.dist_thr == 1.10
    # Check that arguments were mapped correctly (e.g. video_names -> bag_names)
    assert dataset.bag_names == ["vid1"]
    assert hasattr(dataset, "_load_bag")

def test_load_bag_coords(mock_video_data):
    dataset = VideoClassificationDataset(**mock_video_data)
    bag = dataset._load_bag(mock_video_data["video_names"][0])

    assert "coords" in bag
    assert isinstance(bag["coords"], np.ndarray)
    
    # Coords should be shape (N, 1)
    assert bag["coords"].shape == (bag["X"].shape[0], 1)
    
    # Coords should be sequential integers [0, 1, 2, 3, 4]
    assert np.array_equal(bag["coords"].flatten(), np.arange(bag["X"].shape[0]))

def test_load_bag_contents(mock_video_data):
    """Test that all requested keys (X, Y, y_inst) are loaded correctly."""
    dataset = VideoClassificationDataset(**mock_video_data)
    bag = dataset._load_bag(mock_video_data["video_names"][0])

    assert "X" in bag
    assert bag["X"].shape == (5, 64)
    
    assert "Y" in bag
    
    assert "y_inst" in bag
    assert bag["y_inst"].shape[0] == 5