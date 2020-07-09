
import torch
from torch.nn import MSELoss
from open_gns.simulator import Simulator
import pytest


def test_simulator_rollout(tiny_dataset, device):
    # Perform rollout using the simulator
    mse = MSELoss()
    rollout = tiny_dataset[0:143]
    data = rollout[0]
    data = data.to(device)
    v = data.x[:, 5:20]
    sim = Simulator(positions=data.pos, velocities=v, properties=data.x[:, 3:5], device=device)
    assert torch.all(sim.data.edge_index == data.edge_index)
    assert torch.all(sim.data.pos == data.pos)
    positions = []
    accelerations = []
    velocities = []
    acc_gt = []
    pos_gt = []
    vel_gt = []
    for i, data in enumerate(rollout[1:]):
        pos_gt.append(data.pos)
        acc_gt.append(data.y)
        vel_gt.append(data.x[:, 17:20])
        data = data.to(device)
        # Predict
        pos, vel, acc = sim.step()
        positions.append(pos.detach().cpu())
        accelerations.append(acc.detach().cpu())
        velocities.append(vel.detach().cpu())
        # Compare against dataset
        loss = mse(acc, data.y)
