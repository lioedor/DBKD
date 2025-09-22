import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile, os
from pathlib import Path
from collections import defaultdict, Counter


logger = logging.getLogger()

def load_pos_weight(path, device, num_labels: int | None = None) -> torch.Tensor:
    w = np.load(path, allow_pickle=True)
    if w.ndim != 1:
        w = w.reshape(-1)
    w = w.astype(np.float32, copy=False)
    return torch.from_numpy(w).to(device)


def logitsToPred(logits, threshold):
    probabilities = torch.sigmoid(logits)
    predicted_labels = (probabilities >= threshold).float()
    return predicted_labels


class BHLLoss(nn.Module):
    def __init__(self, config):
        super(BHLLoss, self).__init__()
        self.hmidx = np.load(config["hmidx"], allow_pickle=True).item()
        comatrix: np.ndarray = np.load(config["comatrix"])
        comatrix = np.nan_to_num(comatrix, nan=0.0, posinf=0.0, neginf=0.0)
        device = torch.device(config["device"])
        self.comatrix: torch.Tensor = torch.from_numpy(comatrix).float().to(device)
        self.comatrix.requires_grad_(False)
        self.threshold = config["cls_threshold"]
        self.c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        self.lamda = config["lambda"]
        self.gamma = config["gamma"]
        self.beta = config["beta"]
        self.hire_margin = config["hire_margin"]

        self.hier_level_idx = np.load(config["hier_level_idx"], allow_pickle=True)
        self.num_labels = len(self.c2idx)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        with torch.no_grad():
            A = self.comatrix
            deg = A.sum(dim=1)
            d_inv_sqrt = torch.zeros_like(deg)
            mask = deg > 0
            d_inv_sqrt[mask] = deg[mask].pow(-0.5)
            # D^{-1/2} A D^{-1/2}
            normA = (d_inv_sqrt.unsqueeze(1) * A) * d_inv_sqrt.unsqueeze(0)
            L = torch.eye(A.size(0), device=device, dtype=A.dtype) - normA
            L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
            self.register_buffer("L", L)

        par_indices = []
        for idx in range(self.num_labels):
            pidx = self.hmidx.get(idx, self.num_labels)
            par_indices.append(pidx if pidx < self.num_labels else idx)
        self.register_buffer("parent_idx",
            torch.tensor(par_indices, dtype=torch.long, device=device))
        
        ar = torch.arange(self.num_labels, device=device)
        self.register_buffer('is_root', (self.parent_idx == ar))

        pos_weight = load_pos_weight(config["pos_weight"], device=device, num_labels=self.num_labels)
        self.register_buffer('pos_weight', pos_weight)

        logger.info(
            "len(hmidx): {}, shape(comatrix): {}, num_labels: {}".format(
                len(self.hmidx), self.comatrix.shape, self.num_labels
            )
        )
    
    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_parent = p.index_select(1, self.parent_idx)
        y_parent = y.index_select(1, self.parent_idx)

        bce_elem = F.binary_cross_entropy_with_logits(
                logits, y, reduction='none', pos_weight=self.pos_weight
            )
        not_root = (~self.is_root).unsqueeze(0) 
        diff_pos = (p > self.threshold) & (p_parent < self.threshold) & not_root
        w_diff = 1.0 + self.gamma * diff_pos.float()

        hier_mask = ((p_parent > self.threshold) | y_parent.bool() | y.bool()) & not_root
        hier_mask = hier_mask or y or y_parent
        w_diff += self.beta * hier_mask.float()

        smooth = torch.einsum('bc,cd,bd->b', p, self.L, p)
        smooth = torch.nan_to_num(smooth, nan=0.0, posinf=1e4, neginf=0.0)
        smooth = smooth / float(logits.size(1))

        loss = (1 - self.lamda) * (bce_elem * w_diff).mean(dim=1) + self.lamda * smooth
        return loss.mean()
