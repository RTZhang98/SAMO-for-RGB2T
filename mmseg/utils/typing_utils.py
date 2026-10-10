from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.config import ConfigDict

from mmseg.structures import SegDataSample


ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

MultiConfig = Union[ConfigType, Sequence[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

SampleList = Sequence[SegDataSample]
OptSampleList = Optional[SampleList]


TensorDict = Dict[str, torch.Tensor]
TensorList = Sequence[torch.Tensor]

ForwardResults = Union[Dict[str, torch.Tensor], List[SegDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
