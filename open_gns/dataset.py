
import numpy as np
import h5py
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import RadiusGraph

R=0.08 # Connectivity radius $R$

class GNSDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, split='train'):
        super(GNSDataset, self).__init__(root, transform, pre_transform)
        if split == 'train':
            idx = 0
        elif split == 'validation':
            idx = 1
        elif split == 'test':
            idx = 2
        else:
            raise ValueError(f'Split {split} not found.')
        self.data, self.slices = torch.load(self.processed_paths[idx])
        
    @property
    def raw_file_names(self):
        return ['box_bath.hdf5']
    
    @property
    def processed_file_names(self):
        return [f'box_bath_{split}.pt' for split in ['train', 'val', 'test']]
    
    def process_rollout(self, positions):
        data_list = []
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
            y = torch.tensor(accelerations[t]).float()
            data = Data(x=torch.tensor(x).float(), y=y, pos=torch.as_tensor(positions[t]))
            # print(f'Step {t}:', data)
            # Apply pre-transform to get edges
            calculate_edges = self.pre_transform or RadiusGraph(R)
            data = calculate_edges(data)
            data_list.append(data)
        return data_list
    
    def process(self):
        # Read all positions & transform into features
        f = h5py.File(self.raw_file_names[0],'r')
        rollouts = {
            'train': range(120),
            'val': range(120,135),
            'test': range(135,150)
        }
        for i, split in enumerate(rollouts):
            data_list = []
            for rollout in rollouts[split]:
                print('Rollout:', rollout)
                positions = np.array(f.get(f'rollouts/{rollout}/positions'))
                try:
                    data_list += self.process_rollout(positions)
                except Exception as e:
                    print(f'Bad rollout? {rollout}: {e}')
            print(f'Saving {len(data_list)} steps to {self.processed_paths[i]}')
            torch.save(self.collate(data_list), self.processed_paths[i])
