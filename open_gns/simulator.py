
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
from open_gns.models import EncodeProcessDecode

class Simulator():
    def __init__(self, *, positions, properties, velocities=None, device=None, R=0.08):
        # initialize the model
        self.R = R
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('checkpoint_9_7.330730671527333e-07.pt')
        input_size = 25
        model = EncodeProcessDecode(input_size).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model
        self.positions = positions.to(device)
        self.properties = properties.to(device)
        self.velocities = velocities if velocities is not None else torch.zeros((len(positions), 5*3))
        self.velocities = self.velocities.to(device)
        self.data = self.make_graph(positions, properties, self.velocities)

    
    def make_graph(self, positions, properties, velocities):
        d = torch.stack([
            positions[:,1],       # bottom
            positions[:,0],       # left
            positions[:,2],        # back
            1.2 - positions[:,0], # right
            0.4 - positions[:,2]   # front
        ], dim=1)
        d = torch.clamp(d, min=0, max=self.R)   
        x = torch.cat([positions, properties, velocities, d], 1)
        data = Data(x=x, pos=positions)
        find_edges = RadiusGraph(self.R)
        data = find_edges(data)
        return data
    
    def step(self, pos=None):
        # Predict accelerations
        data = self.data
        if pos is not None:
            data.x[:,:3] = pos
            data.pos = pos
        accelerations_ = self.model(data.x, data.edge_index)
        velocities_ = data.x[:,17:20] + accelerations_ 
        positions_ = data.pos + velocities_
        print('p_t:', data.x[0], data.pos[0])
        print('a_t:', accelerations_[0])
        print('v_t:', data.x[0,17:20])
        print('v_t+1',velocities_[0])
        print('p_t+1', positions_[0])
        # Reconstruct data for next frame
        self.velocities = torch.cat([self.velocities[:,3:], velocities_], 1)
        self.data = self.make_graph(positions_, self.properties, self.velocities)
        return positions_, velocities_, accelerations_
