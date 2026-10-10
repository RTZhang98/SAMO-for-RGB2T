import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class SegVisualizationHook(Hook):


    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:

            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:


        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)
