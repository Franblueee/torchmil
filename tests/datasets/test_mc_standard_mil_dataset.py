import pytest
import torch
from torchmil.datasets import MCStandardMILDataset

# --- Fixtures ---

@pytest.fixture
def default_dataset():
    """Returns a standard dataset for general testing."""
    # We seed torch here to ensure the fixture itself is reproducible across test runs
    torch.manual_seed(42)
    return MCStandardMILDataset(D=5, num_bags=10, pos_class_prob=0.5, seed=42)

# --- Tests ---

def test_init_properties(default_dataset):
    """Test initialization of attributes and distributions."""
    ds = default_dataset
    assert ds.num_bags == 10
    assert ds.pos_class_prob == 0.5
    assert ds.train is True
    assert len(ds) == 10
    
    # Verify distributions exist
    assert isinstance(ds.pos_distr, list)
    assert len(ds.pos_distr) == 2
    assert isinstance(ds.neg_distr, torch.distributions.Normal)
    assert isinstance(ds.poisoning, torch.distributions.Normal)

def test_len_empty_edge_case():
    """Test edge case for dataset length."""
    dataset_empty = MCStandardMILDataset(D=2, num_bags=0)
    assert len(dataset_empty) == 0

def test_getitem_bounds(default_dataset):
    """Test __getitem__ boundary conditions."""
    ds = default_dataset
    num_bags = len(ds)
    
    # Valid positive index
    _ = ds[0]
    _ = ds[num_bags - 1]
    
    # Valid negative index
    _ = ds[-1]
    
    # Invalid positive index (Explicit raise in code)
    with pytest.raises(IndexError, match="out of range"):
        _ = ds[num_bags]
        
    # Invalid negative index (Implicit raise by list)
    with pytest.raises(IndexError):
        _ = ds[-(num_bags + 1)]

def test_getitem_structure_and_shapes():
    """
    Test that X (features) and y_inst (instance labels) have matching dimensions.
    This covers the stack/view/cat logic in the sample methods.
    """
    D = 4
    # Seeding torch to ensure sampling doesn't hit edge cases (though unlikely here)
    torch.manual_seed(123)
    ds = MCStandardMILDataset(D=D, num_bags=5, seed=123)
    
    for i in range(len(ds)):
        bag = ds[i]
        X = bag["X"]
        y_inst = bag["y_inst"]
        Y = bag["Y"]
        
        # Check types
        assert isinstance(X, torch.Tensor)
        assert isinstance(y_inst, torch.Tensor)
        
        # Check dimensions
        assert X.ndim == 2
        assert X.shape[1] == D
        assert y_inst.ndim == 1
        
        # CRITICAL: Number of instances must match
        assert X.shape[0] == y_inst.shape[0]
        
        # Check label consistency
        if Y.item() == 1:
            # Positive bags must have positive instances (label 1)
            assert (y_inst == 1).sum() > 0

def test_determinism_and_seeding():
    """
    Test that the dataset generation is reproducible.
    
    NOTE: The dataset implementation only seeds NumPy internally. 
    It uses PyTorch for sampling, so we must manually seed PyTorch 
    in the test to guarantee identical bags.
    """
    seed = 999
    
    # Run 1
    torch.manual_seed(seed)
    ds1 = MCStandardMILDataset(D=3, num_bags=10, seed=seed)

    # Run 2
    torch.manual_seed(seed)
    ds2 = MCStandardMILDataset(D=3, num_bags=10, seed=seed)

    # Run 3 (Control: different seed)
    torch.manual_seed(123)
    ds3 = MCStandardMILDataset(D=3, num_bags=10, seed=123)

    # Check Exact Match
    for i in range(10):
        # We check both the data content and the instance labels
        assert torch.equal(ds1[i]["X"], ds2[i]["X"]), f"Bag {i} data mismatch"
        assert torch.equal(ds1[i]["y_inst"], ds2[i]["y_inst"]), f"Bag {i} labels mismatch"
        assert ds1[i]["Y"] == ds2[i]["Y"]

    # Check Mismatch (Sanity check that seeding actually works)
    # The first bag is highly likely to differ
    assert not torch.equal(ds1[0]["X"], ds3[0]["X"])

def test_train_vs_test_poisoning_logic():
    """
    Strictly verify the poisoning logic:
    - Train: Negative bags have poison (label -1).
    - Test: Positive bags have poison (label -1).
    """
    D = 2
    torch.manual_seed(1)
    
    # --- Train Mode ---
    ds_train = MCStandardMILDataset(D=D, num_bags=20, train=True, seed=1)
    
    for i in range(len(ds_train)):
        bag = ds_train[i]
        labels = bag["y_inst"]
        is_positive_bag = bag["Y"].item() == 1
        
        if is_positive_bag:
            # Train Positive: No poison
            assert -1 not in labels
        else:
            # Train Negative: Has poison
            assert -1 in labels
            # Verify poison values: Mean -10.0
            poison_indices = (labels == -1).nonzero(as_tuple=True)[0]
            poison_data = bag["X"][poison_indices]
            # Check values are roughly around -10 (far from 0 or 2)
            assert torch.all(poison_data < -5.0) 

    # --- Test Mode ---
    # Re-seed to ensure consistent generation behavior
    torch.manual_seed(1)
    ds_test = MCStandardMILDataset(D=D, num_bags=20, train=False, seed=1)
    
    for i in range(len(ds_test)):
        bag = ds_test[i]
        labels = bag["y_inst"]
        is_positive_bag = bag["Y"].item() == 1
        
        if is_positive_bag:
            # Test Positive: Has poison
            assert -1 in labels
            poison_indices = (labels == -1).nonzero(as_tuple=True)[0]
            poison_data = bag["X"][poison_indices]
            assert torch.all(poison_data < -5.0)
        else:
            # Test Negative: No poison
            assert -1 not in labels

def test_positive_concept_distribution_logic():
    """
    Verify that positive bags contain data from the positive distributions.
    Positive means are 2.0 and 3.0.
    """
    D = 1
    torch.manual_seed(55)
    # Create a positive bag in Train mode (to avoid poison noise)
    ds = MCStandardMILDataset(D=D, num_bags=1, pos_class_prob=1.0, train=True, seed=55)
    bag = ds[0]
    
    X = bag["X"]
    y_inst = bag["y_inst"]
    
    # Filter for positive instances (label 1)
    pos_instances = X[y_inst == 1]
    
    # Ensure values are strictly positive and reasonably close to means 2.0/3.0
    # (Checking > 1.0 safely excludes the 0.0 negatives and -10.0 poisons)
    assert torch.all(pos_instances > 1.0)
    assert torch.all(pos_instances < 5.0)

def test_bag_class_probability():
    """Verify that pos_class_prob controls the class balance."""
    torch.manual_seed(10)
    
    ds_all_pos = MCStandardMILDataset(D=2, num_bags=10, pos_class_prob=1.0)
    labels_all_pos = [ds_all_pos[i]["Y"].item() for i in range(10)]
    assert all(label == 1 for label in labels_all_pos)

    ds_all_neg = MCStandardMILDataset(D=2, num_bags=10, pos_class_prob=0.0)
    labels_all_neg = [ds_all_neg[i]["Y"].item() for i in range(10)]
    assert all(label == 0 for label in labels_all_neg)