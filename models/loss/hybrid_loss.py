import torch.nn as nn


class HybridLoss(nn.Module):
  
    def __init__(self, alpha_cls, beta_cls, alpha=0.5, beta=0.5, ignore_zero_labels=False):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.alpha_loss = alpha_cls()
        self.beta = beta
        self.beta_loss = beta_cls()
        self.ignore_zero_labels = ignore_zero_labels

    def forward(self, logits, labels):

        if self.ignore_zero_labels:
            # Mask off examples in the batch that had zero-masks
            mask = labels
            for dim in range(logits.dim() - 1, 0, -1):
                mask = mask.sum(dim, keepdim=True)
            mask = mask.clamp(0, 1)
            logits *= mask
            labels *= mask

        alpha_loss = self.alpha_loss(logits, labels)
        beta_loss = self.beta_loss(logits, labels)
        total_loss = self.alpha * alpha_loss + self.beta * beta_loss

        return total_loss
