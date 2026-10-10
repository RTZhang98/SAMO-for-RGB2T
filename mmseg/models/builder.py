import warnings

from mmseg.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):

    warnings.warn('``build_backbone`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return BACKBONES.build(cfg)


def build_neck(cfg):

    warnings.warn('``build_neck`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return NECKS.build(cfg)


def build_head(cfg):

    warnings.warn('``build_head`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return HEADS.build(cfg)


def build_loss(cfg):

    warnings.warn('``build_loss`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):

    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
