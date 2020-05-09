
import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MetaLayer

def make_mlp(input_size, hidden_size=128, output_size=128, layer_norm=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layers = [
        Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size),
    ]
    if layer_norm:
        layers.append(LayerNorm(output_size))
    return Sequential(*layers).to(device)

class EdgeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(EdgeModel, self).__init__()
        self.edge_mlp = make_mlp(input_size, hidden_size)

    def forward(self, src, dest, edge_attr, u, batch):
        features = [src, dest] if edge_attr is None else [src, dest, edge_attr]
        out = torch.cat(features, 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(NodeModel, self).__init__()
        self.node_mlp = make_mlp(input_size, hidden_size)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # TODO: Do we need to combine with edge_attr?
        return self.node_mlp(x)
                
class Encoder(torch.nn.Module):
    def __init__(self, input_size, output_size=128):
        super(Encoder, self).__init__()
        self.encoder = MetaLayer(EdgeModel(2*input_size), NodeModel(input_size))
    
    def forward(self, x, edge_index):
        # TODO: The encoder needs to build the Graph
        # otherwise the graph would need to be pre-built
        return self.encoder(x, edge_index)
