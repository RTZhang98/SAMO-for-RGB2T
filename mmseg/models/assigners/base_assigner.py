from abc import ABCMeta, abstractmethod
from typing import Optional

from mmengine.structures import InstanceData


class BaseAssigner(metaclass=ABCMeta):


    @abstractmethod
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs):
        pass
