import warnings
from pathlib import Path
from typing import Optional, Union

import mmcv
import numpy as np
import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import mkdir_or_exist

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette
from mmseg.visualization import SegLocalVisualizer
from .utils import ImageType, _preprare_data


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):


    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if config.model.type == 'EncoderDecoder':
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
    elif config.model.type == 'MultimodalEncoderDecoder':
        for k, v in config.model.items():
            if isinstance(v, dict) and 'init_cfg' in v:
                config.model[k].init_cfg = None
    config.model.pretrained = None
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmseg'))

    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        dataset_meta = checkpoint.get('meta', {}).get('dataset_meta', None)

        if 'dataset_meta' in checkpoint.get('meta', {}):

            model.dataset_meta = dataset_meta
        elif 'CLASSES' in checkpoint.get('meta', {}):

            classes = checkpoint['meta']['CLASSES']
            palette = checkpoint['meta']['PALETTE']
            model.dataset_meta = {'classes': classes, 'palette': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, classes and palette will be'
                'set according to num_classes ')
            num_classes = model.decode_head.num_classes
            dataset_name = None
            for name in dataset_aliases.keys():
                if len(get_classes(name)) == num_classes:
                    dataset_name = name
                    break
            if dataset_name is None:
                warnings.warn(
                    'No suitable dataset found, use Cityscapes by default')
                dataset_name = 'cityscapes'
            model.dataset_meta = {
                'classes': get_classes(dataset_name),
                'palette': get_palette(dataset_name)
            }
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_model(model: BaseSegmentor,
                    img: ImageType) -> Union[SegDataSample, SampleList]:


    data, is_batch = _preprare_data(img, model)


    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


def show_result_pyplot(model: BaseSegmentor,
                       img: Union[str, np.ndarray],
                       result: SegDataSample,
                       opacity: float = 0.5,
                       title: str = '',
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       wait_time: float = 0,
                       show: bool = True,
                       with_labels: Optional[bool] = True,
                       save_dir=None,
                       out_file=None):


    if hasattr(model, 'module'):
        model = model.module
    if isinstance(img, str):
        image = mmcv.imread(img, channel_order='rgb')
    else:
        image = img
    if save_dir is not None:
        mkdir_or_exist(save_dir)

    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=save_dir,
        alpha=opacity)
    visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])
    visualizer.add_datasample(
        name=title,
        image=image,
        data_sample=result,
        draw_gt=draw_gt,
        draw_pred=draw_pred,
        wait_time=wait_time,
        out_file=out_file,
        show=show,
        with_labels=with_labels)
    vis_img = visualizer.get_image()

    return vis_img
