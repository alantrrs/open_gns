
from open_gns.dataset import GNSDataset
from open_gns.models import EncodeProcessDecode
from open_gns.train import train
from open_gns.simulator import Simulator
import torch
import pytest

@pytest.mark.longrun
def test_train_and_evaluate(device):
    # Train
    train_dataset = GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(0, 2))
    val_dataset = GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(2, 3))
    model = EncodeProcessDecode(train_dataset.num_node_features)
    train(model, train_dataset, val_dataset, num_epochs=2)
    # Save
    torch.save({'model_state_dict': model.state_dict}, 'model.pt')
    # Simulate
    data = val_dataset[0]
    data = data.to(device)
    v = data.x[:, 5:20]
    sim = Simulator(mode_file='model.pt', positions=data.pos, velocities=v, properties=data.x[:, 3:5], device=device)
    positions = []
    accelerations = []
    velocities = []
    acc_gt = []
    pos_gt = []
    vel_gt = []
    for i, data in enumerate(dataset[1:]):
        pos_gt.append(data.pos)
        acc_gt.append(data.y)
        vel_gt.append(data.x[:, 17:20])
        data = data.to(device)
        # Predict
        pos, vel, acc = sim.step()
        positions.append(pos.detach().cpu())
        accelerations.append(acc.detach().cpu())
        velocities.append(vel.detach().cpu())

