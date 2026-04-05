import torch
import torch.nn.functional as F


def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    probs_a = F.softmax(logits_a, dim=1)
    probs_b = F.softmax(logits_b, dim=1)
    return F.mse_loss(probs_a, probs_b)
