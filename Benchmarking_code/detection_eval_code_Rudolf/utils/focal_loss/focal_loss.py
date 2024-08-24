import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
        self.alpha = alpha
        
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha)
            
        if reduction in ['none', 'mean', 'sum']:
            self.reduction = reduction
        else:
            raise ValueError(f"Reduction must be one of 'none', 'mean' or 'sum'. Got: {reduction}.")

    def forward(self, input:torch.Tensor, target:torch.Tensor):
        # pt = F.softmax(predis, dim=1)
        # logpt = torch.log(pt)
        # Numerically more stable:
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)

        cel = - target * logpt

        fcel = (1 - pt)**self.gamma * cel
        
        if self.alpha is not None:
            fcel = fcel * self.alpha.to(input.device)

        if self.reduction == 'none':
            return fcel
        elif self.reduction == 'mean':
            return fcel.mean()
        elif self.reduction == 'sum':
            return torch.sum(fcel, dim=1)
        
    
    def __str__(self) -> str:
        return f"FocalLoss_gamma{self.gamma}"