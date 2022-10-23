import torch
# from torchtyping import TensorType, patch_typeguard
# from typeguard import typechecked


# @typechecked
def batch_product(batch: torch.Tensor, weight: torch.Tensor):
    r"""
    Multiplies each slice of the first dimension of batch by the corresponding scalar in the weight vector
    :param batch: Tensor of size [B, ...].
    :param weight: Tensor of size [B].
    :return: A tensor such that `result[i] = batch[i] * weight[i]`.
    """
    assert batch.size(0) == weight.size(0)
    return (batch.T * weight.T).T
