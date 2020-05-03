# Dataset
import numpy as np
import h5py
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import RadiusGraph

R=0.08 # Connectivity radius $R$

class GNSDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GNSDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['box_bath.hdf5']
    
    @property
    def processed_file_names(self):
        return ['box_bath_samples.pt']
    
    def process(self):
        # TODO: Split into train, test, val
        data_list = []
        # Read all positions & transform into features
        f = h5py.File(self.raw_file_names[0],'r')
        for k in range(1): # TODO: f.get('rollouts').keys():
            # Read positions
            positions = np.array(f.get(f'rollouts/{k}/positions'))
            num_steps = len(positions)
            # Calculate velocities
            velocities = np.concatenate(([np.zeros(positions[0].shape)],
                                        positions[1:] - positions[0:-1]),axis=0)
            # Calculate accelerations
            accelerations = np.concatenate(([np.zeros(velocities[0].shape)],
                                        velocities[1:] - velocities[0:-1]),axis=0)
            # Material properties (using one-hot encoding for now)
            m = np.zeros((len(positions[0]), 2))
            m[0:64] = [0,1] # First 64 particles are solid
            m[64:] = [1,0]
            # TODO: Global forces
            # TODO: Distance to walls
            # Drop the first 5 and the last step since we don't have accurate velocities/accelerations
            for t in range(6,num_steps-1):
                x = np.concatenate((positions[t], m, np.concatenate(velocities[t-5:t], axis=1)), axis=1)
                y = torch.tensor(accelerations[t])
                data = Data(x=torch.tensor(x), y=y, pos=torch.as_tensor(positions[t]))
                # Apply pre-transform to get edges
                calculate_edges = self.pre_transform or RadiusGraph(R)
                data = calculate_edges(data)
                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
    
