import torch
import torch.nn.functional as F
from torch import nn

from .clb import ClassifierBranch
from .flb import FeatureBranch
from .stud import MainBranch
from .bhl import BHLLoss


def logits_distillation_loss(student_logits, teacher_logits, temperature):
    T = temperature
    student_prob = nn.functional.sigmoid(student_logits / T)
    teacher_prob = nn.functional.sigmoid(teacher_logits / T)

    # KL 散度损失（使用二进制交叉熵计算）
    kl_loss = nn.functional.binary_cross_entropy(
        student_prob, teacher_prob, reduction="none"
    )
    kl_loss = kl_loss.mean(dim=1).sum() * (T * T)

    return kl_loss


def feat_distillation_loss(student_feat, teacher_feat):
    s_norm = F.normalize(student_feat, p=2, dim=1)
    t_norm = F.normalize(teacher_feat, p=2, dim=1)
    cos_sim = s_norm * t_norm
    l = 1 - cos_sim

    return torch.mean(l, dim=1).sum()


class DBKBFramework(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self._join_train: bool = config["join_train"]
        self.temperature = config["temperature"]

        self._mlb = MainBranch(config)
        self.s_bce = nn.BCEWithLogitsLoss()

        self._clb = ClassifierBranch(config)
        self.bhl = BHLLoss(config)

        self._flb = FeatureBranch(config)
        self.f_bce = nn.BCEWithLogitsLoss()

        if not self._join_train:
            self._clb.load()
            self._flb.load()

    def forward(self, x, y, adj_param):
        (h_f, logits_f) = self._flb(x)
        logits_c = self._clb(x)
        h_s, logits_s = self._mlb(x)

        ll = logits_distillation_loss(logits_s, logits_c, self.temperature)  # []
        fl = feat_distillation_loss(torch.cat([h_s, h_s], dim=-1), h_f)  # []
        distillation_loss = adj_param * fl + (1 - adj_param) * ll
        bce_loss = self.s_bce(logits_s, y)  # []

        total_loss = distillation_loss + bce_loss

        if self._join_train:
            total_loss += self.f_bce(logits_f, y)
            total_loss += self.bhl(logits_c, y)

        return logits_s, total_loss
