import os.path as osp
import shutil
from collections import OrderedDict
from typing import Dict, Optional, Sequence

try:

    import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval
    import cityscapesscripts.helpers.labels as CSLabels
except ImportError:
    CSLabels = None
    CSEval = None

import numpy as np
from mmengine.dist import is_main_process, master_only
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image

from mmseg.registry import METRICS


@METRICS.register_module()
class CityscapesMetric(BaseMetric):


    def __init__(self,
                 output_dir: str,
                 ignore_index: int = 255,
                 format_only: bool = False,
                 keep_results: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if CSEval is None:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        self.output_dir = output_dir
        self.ignore_index = ignore_index

        self.format_only = format_only
        if format_only:
            assert keep_results, (
                'When format_only is True, the results must be keep, please '
                f'set keep_results as True, but got {keep_results}')
        self.keep_results = keep_results
        self.prefix = prefix
        if is_main_process():
            mkdir_or_exist(self.output_dir)

    @master_only
    def __del__(self) -> None:

        if not self.keep_results:
            shutil.rmtree(self.output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:


        mkdir_or_exist(self.output_dir)

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'][0].cpu().numpy()


            pred_label = self._convert_to_label_id(pred_label)
            basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
            png_filename = osp.abspath(
                osp.join(self.output_dir, f'{basename}.png'))
            output = Image.fromarray(pred_label.astype(np.uint8)).convert('P')
            output.save(png_filename)
            if self.format_only:

                gt_filename = ''
            else:


                gt_filename = data_sample['seg_map_path'].replace(
                    'labelTrainIds.png', 'labelIds.png')
            self.results.append((png_filename, gt_filename))

    def compute_metrics(self, results: list) -> Dict[str, float]:


        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        eval_results = dict()
        print_log(
            f'Evaluating results under {self.output_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(self.output_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        pred_list, gt_list = zip(*results)
        metric = dict()
        eval_results.update(
            CSEval.evaluateImgLists(pred_list, gt_list, CSEval.args))
        metric['averageScoreCategories'] = eval_results[
            'averageScoreCategories']
        metric['averageScoreInstCategories'] = eval_results[
            'averageScoreInstCategories']
        return metric

    @staticmethod
    def _convert_to_label_id(result):

        if isinstance(result, str):
            result = np.load(result)
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy
