import torch


def hard_pixel_mining(loss_map: torch.Tensor, top_k: float) -> torch.Tensor:
    if top_k <= 0 or top_k > 1:
        return loss_map.mean()
    flat = loss_map.view(-1)
    k = max(1, int(flat.numel() * top_k))
    values, _ = torch.topk(flat, k)
    return values.mean()
