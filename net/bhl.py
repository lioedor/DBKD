import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger()


def logitsToPred(logits, threshold):
    probabilities = torch.sigmoid(logits)
    predicted_labels = (probabilities >= threshold).float()
    return predicted_labels


class BHLLoss(nn.Module):
    def __init__(self, config):
        super(BHLLoss, self).__init__()
        self.hmidx = np.load(config["hmidx"], allow_pickle=True).item()
        self.comatrix: np.ndarray = np.load(config["comatrix"])
        self.comatrix: torch.Tensor = torch.from_numpy(self.comatrix).float()
        self.comatrix.requires_grad_(False)
        self.threshold = config["cls_threshold"]
        self.c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        self.lamda = config["lambda"]
        self.par_indices = []
        self.hier_level_idx = np.load(config["hier_level_idx"], allow_pickle=True)
        self.num_labels = len(self.c2idx)
        for idx in range(self.num_labels):
            pidx = self.num_labels
            if idx in self.hmidx:
                pidx = self.hmidx[idx]

            self.par_indices.append(pidx if pidx < self.num_labels else idx)

        logger.info(
            "len(hmidx): {}, shape(comatrix): {}, num_labels: {}".format(
                len(self.hmidx), self.comatrix.shape, self.num_labels
            )
        )

    # Forward method to compute the loss
    def forward(self, logits, y):
        y_hat = logitsToPred(logits, self.threshold)
        # p = torch.sigmoid(logits)
        # y_hat = torch.where(p >= self.threshold, torch.tensor(1.0), torch.tensor(0.0))
        # y = y_hat = p = [batch_size, num_labels]
        L_down = 0
        y_hat_down = y_hat
        for l, idxs in enumerate(self.hier_level_idx):
            if l == 0:
                y_eles = y[:, idxs]
                y_hat_eles = y_hat[:, idxs]
                # Compute |element-wise| binary cross-entropy , reduction='none'
                bce_loss = F.binary_cross_entropy(y_hat_eles, y_eles, reduction="none")
                L_down += torch.mean(bce_loss, dim=1)
            else:
                y_eles = y[:, idxs]
                y_hat_eles = y_hat_down[:, idxs]
                y_parent_hat_eles = y_hat_down[:, self.par_indices]
                y_parent_hat_eles = y_parent_hat_eles[:, idxs]
                # follow 'false' parent
                y_hat_eles = y_hat_eles * (y_parent_hat_eles != 0)
                bce_loss = F.binary_cross_entropy(y_hat_eles, y_eles, reduction="none")
                bce_loss = bce_loss * y_hat_eles
                bce_loss = torch.mean(bce_loss, dim=1)
                L_down += bce_loss
                y_hat_down[:, idxs] = y_hat_eles

        bce_loss = F.binary_cross_entropy(y_hat, y, reduction="none")
        y_p_hat = y_hat[:, self.par_indices]  # shape same as y_hat_eles
        theta = torch.logical_xor(y_p_hat, y_hat)
        L_up = torch.mean(bce_loss * theta, dim=1)

        p = logits
        p_norm = F.normalize(p, p=2, dim=1)
        com_norm = F.normalize(self.comatrix, p=2, dim=1)
        cos_sim = torch.matmul(p_norm, com_norm)
        d = 1 - cos_sim
        rol = torch.sigmoid(p)
        reg = torch.mean(d * rol, dim=1)

        batch_loss = (1 - self.lamda) * (L_down + L_up) + self.lamda * reg
        return batch_loss.mean()


if __name__ == "__main__":
    import torchsummary

    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    model = BHLLoss(config)
    print(model)
    input = (10441,)  # batch_size, word_length, vector_size
    # permute -> [1, 100, 200], [1, 100, 30]
    torchsummary.summary(model, input)
