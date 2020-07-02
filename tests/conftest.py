
import pytest
from open_gns.dataset import GNSDataset


@pytest.fixture
def tiny_dataset():
    return GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(1,))
