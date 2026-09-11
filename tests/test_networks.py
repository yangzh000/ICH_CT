import numpy as np
import pytest
import torch

from hemorrhage.networks import build_network, complete_batches, set_seed


@pytest.mark.parametrize("name", ["resnet34", "swin"])
def test_3d_forward_backward(name, config):
    set_seed(29)
    torch.set_num_threads(2)
    model = build_network(name, config["deep"])
    inputs = torch.randn(2, 1, 32, 32, 32)
    output = model(inputs)
    assert output.shape == (2, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:, 0], torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_network_batches_keep_every_patient():
    for length in [2, 3, 8, 9, 17, 33]:
        batches = complete_batches(length, 8, torch.Generator().manual_seed(12))
        assert min(map(len, batches)) >= 2
        assert sorted(index for batch in batches for index in batch) == list(range(length))
