
import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MetaLayer

class EdgeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            LayerNorm(hidden_size)
        )
    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest])
        print(f'edge out: {out.size()}')
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(NodeModel, self).__init__()
        self.node_mlp = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            LayerNorm(hidden_size)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.mlp(x)
                
class Encoder(torch.nn.Module):
    def __init__(self, input_size, output_size=128):
        super(Encoder, self).__init__()
        self.encoder = MetaLayer(EdgeModel(2*input_size), NodeModel(input_size))
    
    def forward(self, x, edge_index):
        # TODO: The encoder needs to build the Graph
        # otherwise the graph would need to be pre-built
        print(x, edge_index)
        return self.encoder(x, edge_index)
