import torch.nn as nn
import torch.nn.functional as F

class PartialFocalCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, alpha=1.0, gamma=2.0):
        super(PartialFocalCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        batch_size, num_classes, height, width = outputs.size()
        outputs = outputs.view(batch_size, num_classes, -1)
        targets = targets.view(batch_size, -1)
        targets = targets.long()
        valid_mask = targets != self.ignore_index
        valid_mask_flat = valid_mask.view(-1)
        outputs_flat = outputs.view(-1, num_classes)
        valid_outputs = outputs_flat[valid_mask_flat]
        valid_targets = targets.view(-1)[valid_mask_flat]
        if valid_outputs.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)
        log_probs = F.log_softmax(valid_outputs, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        loss = F.nll_loss(log_probs, valid_targets, reduction='none')
        focal_loss = (focal_weight.gather(1, valid_targets.unsqueeze(1)) * loss).mean()
        return focal_loss