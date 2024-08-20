import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleInteraction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w = nn.Linear(config["enc_dim"], 1)

    def forward(self, H):
        scores = self.w(H)
        weights = torch.softmax(scores, dim=1)
        vec = torch.sum(weights * H, dim=1)
        return vec


class MutalInteraction(nn.Module):
    def __init__(self, config):
        super().__init__()

        label_dim = config["label_dim"]
        enc_dim = config["enc_dim"]

        self.w_t = nn.Linear(enc_dim, 1)
        self.w_l = nn.Linear(label_dim, 1)

    def forward(self, H, E):
        E_transformed = F.gelu(self.w_l(E))
        H_transformed = F.gelu(self.w_t(H))

        E_transformed = E_transformed.unsqueeze(0).unsqueeze(2)
        H_transformed = H_transformed.unsqueeze(1)

        beta_l2d = H_transformed * E_transformed
        beta_l2d = torch.sum(beta_l2d, dim=-1)

        weight = torch.softmax(beta_l2d, dim=-1).unsqueeze(-1)

        H_tilde = H.unsqueeze(1)
        H_tilde = torch.sum(weight * H_tilde, dim=2)

        h_tilde, _ = torch.max(H_tilde, dim=1)

        h_avg = torch.mean(H, dim=1)
        beta_d2l = F.gelu(self.w_t(h_avg)).unsqueeze(1) * F.gelu(self.w_l(E))
        h_bar = torch.mean(beta_d2l * E.unsqueeze(0), dim=1)

        h_f = torch.cat([h_tilde, h_bar], dim=1)
        return h_f
