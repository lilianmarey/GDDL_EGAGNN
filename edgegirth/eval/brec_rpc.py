"""Official BREC RPC (Reliable Paired Comparison) protocol, Wang & Zhang 2023.

Re-implements the T^2 Hotelling test and per-pair train/eval loop of
GraphPKU/BREC's base/test_BREC.py, generalised to take any model factory. No
similarity-threshold proxy is used anywhere -- every method goes through this
same T^2 test with the same reliability control group.
"""

from __future__ import annotations

import torch
from torch.nn import CosineEmbeddingLoss
from torch_geometric.loader import DataLoader

torch.set_num_threads(1)

EPSILON_CMP = 1e-6


def t2_calculation(model, graphs: list, batch_size: int, device):
    with torch.no_grad():
        loader = DataLoader(graphs, batch_size=batch_size)
        pred_0, pred_1 = [], []
        for data in loader:
            pred = model(data.to(device)).detach()
            pred_0.extend(pred[0::2])
            pred_1.extend(pred[1::2])
        X = torch.cat([x.reshape(1, -1) for x in pred_0], dim=0).T
        Y = torch.cat([x.reshape(1, -1) for x in pred_1], dim=0).T
        D = X - Y
        D_mean = torch.mean(D, dim=1).reshape(-1, 1)
        S = torch.cov(D)
        inv_S = torch.linalg.pinv(S)
        return torch.mm(torch.mm(D_mean.T, inv_S), D_mean).item()


def evaluate_pair(
    model_factory,
    graphs_main: list,
    graphs_reliability: list,
    *,
    epoch: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    threshold: float = 72.34,
    margin: float = 0.0,
    loss_threshold: float = 0.2,
    device="cpu",
    learned: bool = True,
):
    """Runs the RPC pipeline (train + T^2 test + reliability control) for one pair.

    Returns dict(isomorphic_flag, reliable, t2_main, t2_reliability).
    """
    model = model_factory().to(device)

    if learned:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        loss_func = CosineEmbeddingLoss(margin=margin)
        model.train()
        loader = DataLoader(graphs_main, batch_size=batch_size)
        num_relabel = len(graphs_main) // 2
        for _ in range(epoch):
            loss_all = 0.0
            for data in loader:
                optimizer.zero_grad()
                pred = model(data.to(device))
                loss = loss_func(pred[0::2], pred[1::2], torch.tensor([-1] * (len(pred) // 2)).to(device))
                loss.backward()
                optimizer.step()
                loss_all += len(pred) / 2 * loss.item()
            loss_all /= num_relabel
            if loss_all < loss_threshold:
                break
            scheduler.step(loss_all)

    model.eval()
    t2_main = t2_calculation(model, graphs_main, batch_size, device)
    t2_reliability = t2_calculation(model, graphs_reliability, batch_size, device)

    isomorphic_flag = (t2_main > threshold) and not (abs(t2_main - t2_reliability) < EPSILON_CMP)
    reliable = t2_reliability < threshold
    return dict(
        isomorphic_flag=isomorphic_flag, reliable=reliable, t2_main=t2_main, t2_reliability=t2_reliability
    )
