import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict

# Assuming the TensorDict class from above is in the same file or imported.

@pytest.fixture
def simple_td():
    """Provides a simple TensorDict with a 1D batch size."""
    data = {
        'observations': torch.randn(4, 10),
        'actions': torch.randint(0, 5, (4, 1))
    }
    return TensorDict(data, torch.Size([4]))

@pytest.fixture
def td_with_2d_batch():
    """Provides a TensorDict with a 2D batch size."""
    data = {
        'pixels': torch.rand(3, 5, 3, 64, 64), # Batch size is (3, 5)
        'rewards': torch.randn(3, 5)
    }
    return TensorDict(data, torch.Size([3, 5]))



def test_mask_select_1d(simple_td):
    """Tests boolean mask selection on a 1D batch."""
    mask = torch.tensor([True, False, True, False])
    selected_td = simple_td[mask]

    assert selected_td.shape == torch.Size([2])
    # Check that each tensor within the dict is correctly filtered.
    assert torch.equal(selected_td.data['observations'], simple_td.data['observations'][mask])
    assert torch.equal(selected_td.data['actions'], simple_td.data['actions'][mask])

def test_mask_select_2d(td_with_2d_batch):
    """Tests boolean mask selection on a 2D batch."""
    mask = torch.tensor([
        [True, False, True, False, True],
        [False, True, False, True, False],
        [True, True, False, False, False]
    ], dtype=torch.bool)

    selected_td = td_with_2d_batch[mask]

    num_selected = mask.sum().item()
    assert selected_td.shape == torch.Size([int(num_selected)])
    assert selected_td.data['pixels'].shape[0] == num_selected
    assert selected_td.data['rewards'].shape[0] == num_selected
    assert torch.equal(selected_td.data['pixels'], td_with_2d_batch.data['pixels'][mask])
    assert torch.equal(selected_td.data['rewards'], td_with_2d_batch.data['rewards'][mask])

def test_tensors_with_extra_dims(simple_td):
    """Tests that tensors with more dimensions than the batch size are handled correctly."""
    mask = torch.tensor([True, True, False, False])
    selected_td = simple_td[mask]

    assert selected_td.shape == torch.Size([2])
    assert selected_td.data['observations'].shape == (2, 10)
    assert selected_td.data['actions'].shape == (2, 1)


def test_all_true_mask(simple_td):
    """Tests selection with a mask that selects everything."""
    mask = torch.ones(4, dtype=torch.bool)
    selected_td = simple_td[mask]

    assert selected_td.shape == simple_td.shape
    # The resulting TensorDict should be equal to the original.
    assert torch.equal(selected_td.data['observations'], simple_td.data['observations'])
    assert torch.equal(selected_td.data['actions'], simple_td.data['actions'])


def test_mask_shape_mismatch(simple_td):
    """Tests that an incorrectly shaped mask raises an IndexError."""
    # Mask is too short
    with pytest.raises(IndexError, match="The shape of the mask.*does not match the shape of the indexed tensor"):
        simple_td[torch.tensor([True, False])]

    # Mask has wrong number of dimensions
    with pytest.raises(IndexError, match="The shape of the mask.*does not match the shape of the indexed tensor"):
        simple_td[torch.ones(4, 1, dtype=torch.bool)]


def test_initialization_with_inconsistent_tensor_shape():
    """Tests that the constructor raises an error for inconsistent tensor shapes."""
    with pytest.raises(ValueError, match="Shape mismatch at.*is not compatible with the TensorDict's batch shape"):
        TensorDict(
            {'a': torch.randn(4, 5), 'b': torch.randn(3, 5)},
            shape=torch.Size([4])
        )
