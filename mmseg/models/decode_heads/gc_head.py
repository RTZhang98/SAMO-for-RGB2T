import torch
from mmcv.cnn import ContextBlock

from mmseg.registry import MODELS
from .fcn_head import FCNHead


@MODELS.register_module()
class GCHead(FCNHead):


    def __init__(self,
                 ratio=1 / 4.,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 **kwargs):
        super().__init__(num_convs=2, **kwargs)
        self.ratio = ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.gc_block = ContextBlock(
            in_channels=self.channels,
            ratio=self.ratio,
            pooling_type=self.pooling_type,
            fusion_types=self.fusion_types)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.gc_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
