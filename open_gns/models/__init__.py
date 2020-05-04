import torch
from open_gns.models.encoder import Encoder
from open_gns.models.processor import Processor
from open_gns.models.decoder import Decoder

class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(EncodeProcessDecode, self).__init__()
        self.encoder = Encoder(input_size)
        self.processor = Processor(hidden_size)
        self.decoder = Decoder(hidden_size)
    
    def forward(self, x, edge_index):
        x, edge_attr, _ = self.encoder(x, edge_index)
        x, edge_attr, _ = self.processor(x, edge_index, edge_attr)
        return self.decoder(x)
