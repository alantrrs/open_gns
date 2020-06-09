
import numpy as np
import h5py
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import RadiusGraph

R=0.08 # Connectivity radius $R$

class GNSDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, split='train'):
        if split == 'train':
            self.split_idx = 0
        elif split == 'validation':
            self.split_idx = 1
        elif split == 'test':
            self.split_idx = 2
        else:
            raise ValueError(f'Split {split} not found.')
        super(GNSDataset, self).__init__(root, transform, pre_transform)
        self.data_file = h5py.File(self.processed_paths[0], 'r')
        
    @property
    def raw_file_names(self):
        return [f'{self.root}/box_bath.hdf5']
    
    @property
    def processed_file_names(self):
        splits = ('train', 'val', 'test')
        return [f'box_bath_{splits[self.split_idx]}.hdf5']
    
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
        # Drop the first 5 and the last step since we don't have accurate velocities/accelerations
        for t in range(6,num_steps-1):
            # Distance to the 5 walls
            d = np.stack([
                positions[t][:,1],       # bottom
                positions[t][:,0],       # left
                positions[t][:,2],        # back
                1.2 - positions[t][:,0], # right
                0.4 - positions[t][:,2]   # front
            ], axis=1)
            d = np.clip(d, 0, R)
            x = np.concatenate((positions[t], m, np.concatenate(velocities[t-5:t], axis=1), d), axis=1)
            y = torch.tensor(accelerations[t]).float()
            data = Data(x=torch.tensor(x).float(), y=y, pos=torch.as_tensor(positions[t]))
            # print(f'Step {t}:', data)
            # Apply pre-transform to get edges
            calculate_edges = self.pre_transform or RadiusGraph(R)
            data = calculate_edges(data)
            data_list.append(data)
        return data_list
    
    def save_data_to_hdf5(self, output_file, data_list, start_index=0):
        for i, data in enumerate(data_list):
            idx = start_index + i
            for k in ['x', 'edge_index', 'pos', 'y']:
                output_file.create_dataset(f'data/{idx}/{k}', data=getattr(data, k), compression='gzip')
        
    def len(self):
        return len(self.data_file.get('data').keys())
    
    def get(self, index):
        d = self.data_file[f'data/{index}']
        data = Data(
            x=torch.as_tensor(np.array(d['x'])),
            edge_index=torch.as_tensor(np.array(d['edge_index'])),
            pos=torch.as_tensor(np.array(d['pos'])),
            y=torch.as_tensor(np.array(d['y'])))
        return data
    

    def process(self):
        # Read all positions & transform into features
        f = h5py.File(self.raw_file_names[0],'r')
        out = h5py.File(self.processed_paths[0], 'w')
        rollouts = [range(1000), range(1000,1100), range(1100,1200)]
        start_index = 0
        for rollout in rollouts[self.split_idx]:
            print('Rollout:', rollout)
            positions = np.array(f.get(f'rollouts/{rollout}/positions'))
            try:
                data_list = self.process_rollout(positions)
                self.save_data_to_hdf5(out, data_list, start_index=start_index)
                start_index += len(data_list)
            except Exception as e:
                print(f'Bad rollout? {rollout}: {e}')
        out.close()
