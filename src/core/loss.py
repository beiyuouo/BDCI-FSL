import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_criterion(cfg, loss_type=None):
    if loss_type is None:
        loss_type = cfg.loss
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "focal":
        criterion = FocalLoss(cfg.focal.alpha, cfg.focal.gamma)
    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    return criterion
