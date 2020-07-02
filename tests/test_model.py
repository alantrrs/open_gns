
import pytest
from open_gns.models.encoder import RelativeEncoder
from open_gns.models.processor import Processor
from open_gns.models.encoder import Encoder


def test_encoder_output_is_the_righ_size(tiny_dataset):
    input_size = tiny_dataset.num_node_features
    data = tiny_dataset[0]
    encoder = RelativeEncoder(input_size)
    x, edge_attr, u = encoder(data.x, data.edge_index)
    assert x.size() == (len(data.x), 128)

@pytest.mark.skip
def test_processor():
    pass

@pytest.mark.skip
def test_encoder():
    pass
