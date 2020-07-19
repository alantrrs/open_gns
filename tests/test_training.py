
from open_gns.train import train
from open_gns.models import EncodeProcessDecode
from open_gns.dataset import GNSDataset

def test_training():
    train_dataset = GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(0, 2))
    val_dataset = GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(2, 3))
    model = EncodeProcessDecode(train_dataset.num_node_features)
    train(model, train_dataset, val_dataset, num_epochs=2)
