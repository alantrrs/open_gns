
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
from open_gns.models import EncodeProcessDecode

class Simulator():
    def __init__(self, *, positions, properties, velocities=None, device=None, R=0.08):
        # initialize the model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('checkpoint_1_5.916705061520588e-06.pt')
        input_size=20
        model = EncodeProcessDecode(input_size).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.find_edges = RadiusGraph(R)
        self.model = model
        self.positions = positions.to(device)
        self.properties = properties.to(device)
        self.velocities = velocities if velocities is not None else torch.zeros((len(positions), 5*3))
        self.velocities = self.velocities.to(device)
        self.data = self.make_graph(positions, properties, self.velocities)

    
    def make_graph(self, positions, properties, velocities):
        x = torch.cat([positions, properties, velocities], 1)
        data = Data(x=x, pos=positions)
        data = self.find_edges(data)
        return data
    
    def step(self):
        # Predict accelerations
        data = self.data
        accelerations_ = self.model(data.x, data.edge_index)
        prev_velocities = data.x[:,-3:]
        velocities_ = data.x[:,-3:] + accelerations_
        positions_ = data.pos + velocities_
        # Reconstruct data for next frame
        self.velocities = torch.cat([self.velocities[:,3:], velocities_], 1)
        self.data = self.make_graph(positions_, self.properties, self.velocities)
        return positions_, velocities_, accelerations_
