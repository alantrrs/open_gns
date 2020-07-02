
from open_gns.dataset import GNSDataset


def test_dataset():
    dataset = GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(0,10))
    assert dataset.num_node_features == 25
