import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=probs.shape[-1]).float()

        # compute the negative expected log likelihood
        loss = -targets_one_hot * log_probs

        # apply the Focal Loss
        weights = (1 - probs) ** self.gamma
        if self.alpha is not None:
            weights = self.alpha * targets_one_hot * weights + (1 - self.alpha) * weights
        loss = (weights * loss).sum()

        return loss