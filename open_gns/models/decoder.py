
import torch
from open_gns.models.encoder import make_mlp

class Decoder(torch.nn.Module):
    def __init__(self, input_size):
        super(Decoder, self).__init__()
        self.decoder = make_mlp(input_size, output_size=1, layer_norm=False)
    
    def forward(self, x):
        return self.decoder(x)

