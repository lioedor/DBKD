import numpy as np
import torch.nn as nn

from .doc_encoder import DocEncoder
from .man import SimpleInteraction
from .classifier import Classifier


class MainBranch(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.doc = DocEncoder(config)
        self.man = SimpleInteraction(config)

        num_feats = config["enc_dim"]
        c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        num_cls = len(c2idx)
        self.cls = Classifier(num_feats, num_cls, config["droprate"])

    def forward(self, x):
        H = self.doc(x)
        h = self.man(H)
        # H = torch.flatten(H, start_dim=1)
        p = self.cls(h)
        return h, p
