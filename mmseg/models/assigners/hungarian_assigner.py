from typing import List, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast

from mmseg.registry import TASK_UTILS
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class HungarianAssigner(BaseAssigner):


    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self, pred_instances: InstanceData, gt_instances: InstanceData,
               **kwargs):


        cost_list = []
        with autocast(enabled=False):
            for match_cost in self.match_costs:
                cost = match_cost(
                    pred_instances=pred_instances, gt_instances=gt_instances)
                cost_list.append(cost)
            cost = torch.stack(cost_list).sum(dim=0)

        device = cost.device

        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_quiery_inds, matched_label_inds = linear_sum_assignment(cost)
        matched_quiery_inds = torch.from_numpy(matched_quiery_inds).to(device)
        matched_label_inds = torch.from_numpy(matched_label_inds).to(device)

        return matched_quiery_inds, matched_label_inds
