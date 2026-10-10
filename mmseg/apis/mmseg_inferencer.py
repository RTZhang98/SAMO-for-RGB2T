import os.path as osp
import warnings
from typing import List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from PIL import Image

from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, get_classes, get_palette
from mmseg.visualization import SegLocalVisualizer

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[SegDataSample, SampleList]


class MMSegInferencer(BaseInferencer):


    preprocess_kwargs: set = set()
    forward_kwargs: set = {'mode', 'out_dir'}
    visualize_kwargs: set = {
        'show', 'wait_time', 'img_out_dir', 'opacity', 'return_vis',
        'with_labels'
    }
    postprocess_kwargs: set = {'pred_out_dir', 'return_datasample'}

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 classes: Optional[Union[str, List]] = None,
                 palette: Optional[Union[str, List]] = None,
                 dataset_name: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmseg') -> None:


        self.num_visualized_imgs = 0
        self.num_pred_imgs = 0
        init_default_scope(scope if scope else 'mmseg')
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

        if device == 'cpu' or not torch.cuda.is_available():
            self.model = revert_sync_batchnorm(self.model)

        assert isinstance(self.visualizer, SegLocalVisualizer)
        self.visualizer.set_dataset_meta(classes, palette, dataset_name)

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:


        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})

            if 'dataset_meta' in checkpoint_meta:

                model.dataset_meta = {
                    'classes': checkpoint_meta['dataset_meta'].get('classes'),
                    'palette': checkpoint_meta['dataset_meta'].get('palette')
                }
            elif 'CLASSES' in checkpoint_meta:

                classes = checkpoint_meta['CLASSES']
                palette = checkpoint_meta.get('PALETTE', None)
                model.dataset_meta = {'classes': classes, 'palette': palette}
            else:
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use classes of Cityscapes by '
                    'default.')
                model.dataset_meta = {
                    'classes': get_classes('cityscapes'),
                    'palette': get_palette('cityscapes')
                }
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            warnings.warn(
                'weights is None, use cityscapes classes by default.')
            model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 out_dir: str = '',
                 img_out_dir: str = 'vis',
                 pred_out_dir: str = 'pred',
                 **kwargs) -> dict:


        if out_dir != '':
            pred_out_dir = osp.join(out_dir, pred_out_dir)
            img_out_dir = osp.join(out_dir, img_out_dir)
        else:
            pred_out_dir = ''
            img_out_dir = ''

        return super().__call__(
            inputs=inputs,
            return_datasamples=return_datasamples,
            batch_size=batch_size,
            show=show,
            wait_time=wait_time,
            img_out_dir=img_out_dir,
            pred_out_dir=pred_out_dir,
            return_vis=return_vis,
            **kwargs)

    def visualize(self,
                  inputs: list,
                  preds: List[dict],
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  img_out_dir: str = '',
                  opacity: float = 0.8,
                  with_labels: Optional[bool] = True) -> List[np.ndarray]:


        if not show and img_out_dir == '' and not return_vis:
            return None
        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        self.visualizer.set_dataset_meta(**self.model.dataset_meta)
        self.visualizer.alpha = opacity

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8) + '_vis'
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type:'
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != ''\
                else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=True,
                out_file=out_file,
                with_labels=with_labels)
            if return_vis:
                results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results if return_vis else None

    def postprocess(self,
                    preds: PredType,
                    visualization: List[np.ndarray],
                    return_datasample: bool = False,
                    pred_out_dir: str = '') -> dict:


        if return_datasample:
            if len(preds) == 1:
                return preds[0]
            else:
                return preds

        results_dict = {}

        results_dict['predictions'] = []
        results_dict['visualization'] = []

        for i, pred in enumerate(preds):
            pred_data = dict()
            if 'pred_sem_seg' in pred.keys():
                pred_data['sem_seg'] = pred.pred_sem_seg.numpy().data[0]
            elif 'pred_depth_map' in pred.keys():
                pred_data['depth_map'] = pred.pred_depth_map.numpy().data[0]

            if visualization is not None:
                vis = visualization[i]
                results_dict['visualization'].append(vis)
            if pred_out_dir != '':
                mmengine.mkdir_or_exist(pred_out_dir)
                for key, data in pred_data.items():
                    post_fix = '_pred.png' if key == 'sem_seg' else '_pred.npy'
                    img_name = str(self.num_pred_imgs).zfill(8) + post_fix
                    img_path = osp.join(pred_out_dir, img_name)
                    if key == 'sem_seg':
                        output = Image.fromarray(data.astype(np.uint8))
                        output.save(img_path)
                    else:
                        np.save(img_path, data)
            pred_data = next(iter(pred_data.values()))
            results_dict['predictions'].append(pred_data)
            self.num_pred_imgs += 1

        if len(results_dict['predictions']) == 1:
            results_dict['predictions'] = results_dict['predictions'][0]
            if visualization is not None:
                results_dict['visualization'] = \
                    results_dict['visualization'][0]
        return results_dict

    def _init_pipeline(self, cfg: ConfigType) -> Compose:


        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        for transform in ('LoadAnnotations', 'LoadDepthAnnotation'):
            idx = self._get_transform_idx(pipeline_cfg, transform)
            if idx != -1:
                del pipeline_cfg[idx]

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'InferencerLoader'
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:


        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1
