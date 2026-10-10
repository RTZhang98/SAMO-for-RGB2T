import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class KLDivLoss(nn.Module):

    def __init__(self,
                 temperature: float = 1.0,
                 reduction: str = 'mean',
                 loss_name: str = 'loss_kld'):


        assert isinstance(temperature, (float, int)), \
            'Expected temperature to be' \
            f'float or int, but got {temperature.__class__.__name__} instead'
        assert temperature != 0., 'Temperature must not be zero'

        assert reduction in ['mean', 'none', 'sum'], \
            'Reduction must be one of the options ("mean", ' \
            f'"sum", "none"), but got {reduction}'

        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self._loss_name = loss_name

    def forward(self, input: torch.Tensor, target: torch.Tensor):


        assert isinstance(input, torch.Tensor), 'Expected input to' \
            f'be Tensor, but got {input.__class__.__name__} instead'
        assert isinstance(target, torch.Tensor), 'Expected target to' \
            f'be Tensor, but got {target.__class__.__name__} instead'

        assert input.shape == target.shape, 'Input and target ' \
            'must have same shape,' \
            f'but got shapes {input.shape} and {target.shape}'

        input = F.softmax(input / self.temperature, dim=1)
        target = F.softmax(target / self.temperature, dim=1)

        loss = F.kl_div(input, target, reduction='none', log_target=False)
        loss = loss * self.temperature**2

        batch_size = input.shape[0]

        if self.reduction == 'sum':

            loss = loss.view(batch_size, -1)
            return torch.sum(loss, dim=1)

        elif self.reduction == 'mean':

            loss = loss.view(batch_size, -1)
            return torch.mean(loss, dim=1)

        return loss

    @property
    def loss_name(self):


        return self._loss_name
