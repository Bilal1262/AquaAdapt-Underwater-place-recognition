import torch

from torch.nn import functional as F

from aquaadapt.models.projection import ProjectionHead, ResidualAdapterHead


def test_projection_shape_normalization_and_gradients() -> None:
    head = ProjectionHead()
    output = head(torch.randn(4, 384))
    assert output.shape == (4, 256)
    assert torch.allclose(output.norm(dim=1), torch.ones(4), atol=1e-5)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_residual_adapter_starts_as_raw_dino_identity() -> None:
    features = torch.randn(4, 384)
    head = ResidualAdapterHead(adapter_scale=0.25)
    output = head(features)
    assert output.shape == (4, 384)
    assert torch.allclose(output, F.normalize(features, dim=-1), atol=1e-6)
    output.sum().backward()
    assert head.adapter[-1].weight.grad is not None
