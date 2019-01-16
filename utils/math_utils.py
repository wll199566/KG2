import torch

def normalize_tensor(torch_tensor):
    """
    normalize torch tensor.
    Args:
        -torch_tensor: torch tensor to normalize.
    Returns:
        -the L2-normalized tensor. 
    """
    # NOTE: to deal with the case that input tensor is zeros tensor.
    if torch.sum(torch_tensor.squeeze(), dim=0).item() == 0:
        return torch_tensor.detach()
    else:
        torch_tensor_n = torch.norm(torch_tensor, p=2).detach()
        return torch_tensor.div(torch_tensor_n.expand_as(torch_tensor))
