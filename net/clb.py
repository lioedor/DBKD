import os
import yaml
import torch
import torch.nn as nn
import numpy as np

from .doc_encoder import DocEncoder
from .man import SimpleInteraction
from .classifier import Classifier


class ClassifierBranch(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.doc = DocEncoder(config)
        self.man = SimpleInteraction(config)

        num_feats = config["enc_dim"]
        # max_words = config["max_words"]
        c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        num_cls = len(c2idx)
        self.cls = Classifier(num_feats, num_cls, config["droprate"])

        self.persistent_path = config["persistent"]["clas"]

    def forward(self, x):
        H = self.doc(x)
        h = self.man(H)
        # H = torch.flatten(H, start_dim=1)
        p = self.cls(h)
        return p

    def load(self):
        if not os.path.exists(self.persistent_path):
            assert ("not found module params at ", self.persistent_path)

        self.load_state_dict(torch.load(self.persistent_path))
        for param in self.parameters():
            param.requires_grad = False

    def save(self):
        torch.save(self.state_dict(), self.persistent_path)
