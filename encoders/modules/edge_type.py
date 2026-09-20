import torch


def decide_edge_type(vec: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Binary classify each edge by length.

    Zero-length edges (self-loops) get type 1, all others get type 0.
    """
    length = torch.norm(vec, dim=-1)
    edge_type = torch.zeros(length.shape, dtype=torch.long, device=vec.device)
    edge_type[length <= epsilon] = 1
    return edge_type
