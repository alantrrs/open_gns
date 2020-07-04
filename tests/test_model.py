
import pytest
import torch
from open_gns.models.encoder import RelativeEncoder
from open_gns.models.processor import Processor
from open_gns.models.decoder import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_encoder_output_is_the_righ_size(tiny_dataset):
    input_size = tiny_dataset.num_node_features
    data = tiny_dataset[0]
    data = data.to(device)
    encoder = RelativeEncoder(input_size)
    x, edge_attr, u = encoder(data.x, data.edge_index)
    assert x.size() == (len(data.x), 128)
    assert edge_attr.size(1) == 128


def test_encoder_processor_output(tiny_dataset):
    data = tiny_dataset[0]
    data = data.to(device)
    encoder = RelativeEncoder(25)
    x, edge_attr, u = encoder(data.x, data.edge_index)
    processor = Processor(128)   
    x, e, u = processor(x, data.edge_index, edge_attr)
    assert x.size() == (len(x), 128)


def test_decoder_output(tiny_dataset):
    data = tiny_dataset[0]
    data = data.to(device)
    encoder = RelativeEncoder(25)
    x, edge_attr, u = encoder(data.x, data.edge_index)
    processor = Processor(128)   
    x, e, u = processor(x, data.edge_index, edge_attr)
    decoder = Decoder(128)
    y = decoder(x)
    assert y.size(1) == 3
