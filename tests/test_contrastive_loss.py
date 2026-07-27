import torch

from aquaadapt.losses.multipositive_infonce import multi_positive_infonce


def _mask() -> torch.Tensor:
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 1] = mask[1, 0] = True
    mask[2, 3] = mask[3, 2] = True
    return mask


def test_finite_loss_and_gradients_with_repeated_descriptors() -> None:
    descriptors = torch.ones(4, 8, requires_grad=True)
    loss = multi_positive_infonce(descriptors, _mask())
    assert torch.isfinite(loss)
    loss.backward()
    assert descriptors.grad is not None
    assert torch.isfinite(descriptors.grad).all()


def test_aligned_positives_have_lower_loss() -> None:
    aligned = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    misaligned = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    assert multi_positive_infonce(aligned, _mask()) < multi_positive_infonce(misaligned, _mask())


def test_no_positives_is_differentiable_zero() -> None:
    descriptors = torch.randn(3, 4, requires_grad=True)
    loss = multi_positive_infonce(descriptors, torch.zeros(3, 3, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert descriptors.grad is not None

