
import pytest
from open_gns.dataset import GNSDataset
import torch


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def tiny_dataset():
    return GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(1,))
