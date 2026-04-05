import torch
import torch.nn.functional as F
from tqdm import tqdm


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()


def adapt_model(model, loader, cfg, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["tta"].get("lr", 1e-4))
    steps = cfg["tta"].get("steps", 4)
    for _ in range(steps):
        for images in tqdm(loader, desc="tta", leave=False):
            images = images.to(device)
            logits = model(images)
            loss = entropy_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
