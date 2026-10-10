from typing import Optional

from mmengine.optim.scheduler import PolyLR

from mmseg.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class PolyLRRatio(PolyLR):


    def __init__(self, eta_min_ratio: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta_min_ratio = eta_min_ratio

    def _get_value(self):


        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        param_groups_value = []
        for base_value, param_group in zip(self.base_values,
                                           self.optimizer.param_groups):
            eta_min = self.eta_min if self.eta_min_ratio is None else \
                base_value * self.eta_min_ratio
            step_ratio = (1 - 1 /
                          (self.total_iters - self.last_step + 1))**self.power
            step_value = (param_group[self.param_name] -
                          eta_min) * step_ratio + eta_min
            param_groups_value.append(step_value)

        return param_groups_value
