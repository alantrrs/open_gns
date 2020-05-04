
import torch
from torch_geometric.nn import MetaLayer
from open_gns.models.encoder import EdgeModel, NodeModel

class Processor(torch.nn.Module):
    def __init__(self, input_size, output_size=128, M=10):
        super(Processor, self).__init__()
        self.GNs = []
        for i in range(M):
            GN = MetaLayer(EdgeModel(3*input_size), NodeModel(input_size))
            self.GNs.append(GN) 

    def forward(self, x, edge_index, edge_attr):
        for GN in self.GNs:
            # TODO: Concatenate residuals instead?
            # Keep residuals
            node_residual = x
            edge_residual = edge_attr
            # Apply GN
            x, edge_attr, u = GN(x, edge_index, edge_attr)
            # Add residuals
            x = x + node_residual
            edge_attr = edge_attr + edge_residual
        return x, edge_attr, u
