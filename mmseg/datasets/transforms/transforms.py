import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union
from PIL import Image

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmcv.image.geometric import _scale_size
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter
from random import sample

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, GaussianBlur,
    RandomRain, MotionBlur, RandomSunFlare
)

try:
    import albumentations
    from albumentations import Compose
    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False


@TRANSFORMS.register_module()
class ResizeToMultiple(BaseTransform):


    def __init__(self, size_divisor=32, interpolation=None):
        self.size_divisor = size_divisor
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:


        img = results['img']
        img = mmcv.imresize_to_multiple(
            img,
            self.size_divisor,
            scale_factor=1,
            interpolation=self.interpolation
            if self.interpolation else 'bilinear')

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]


        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            gt_seg = mmcv.imresize_to_multiple(
                gt_seg,
                self.size_divisor,
                scale_factor=1,
                interpolation='nearest')
            results[key] = gt_seg

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size_divisor={self.size_divisor}, '
                     f'interpolation={self.interpolation})')
        return repr_str


@TRANSFORMS.register_module()
class Rerange(BaseTransform):


    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, results: dict) -> dict:


        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value

        img = (img - img_min_value) / (img_max_value - img_min_value)

        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@TRANSFORMS.register_module()
class CLAHE(BaseTransform):


    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def transform(self, results: dict) -> dict:


        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, ' \
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):


    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '


        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:


        def generate_crop_bbox(img: np.ndarray) -> tuple:


            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:

            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:


        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:


        img = results['img']
        crop_bbox = self.crop_bbox(results)


        img = self.crop(img, crop_bbox)


        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):


    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    @cache_randomness
    def generate_degree(self):
        return np.random.rand() < self.prob, np.random.uniform(
            min(*self.degree), max(*self.degree))

    def transform(self, results: dict) -> dict:


        rotate, degree = self.generate_degree()
        if rotate:

            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)


            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@TRANSFORMS.register_module()
class RGB2Gray(BaseTransform):


    def __init__(self, grayscale_prob=1.0):

        assert 0.0 <= grayscale_prob <= 1.0, \
            f'grayscale_prob must be in [0, 1], got {grayscale_prob}'
        self.grayscale_prob = grayscale_prob

    def transform(self, results: dict) -> dict:


        img = results['img']

        if random.random() < self.grayscale_prob:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = np.stack([gray, gray, gray], axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(grayscale_prob={self.grayscale_prob})'
        return repr_str


@TRANSFORMS.register_module()
class AdjustGamma(BaseTransform):


    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def transform(self, results: dict) -> dict:


        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'


@TRANSFORMS.register_module()
class SegRescale(BaseTransform):


    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def transform(self, results: dict) -> dict:


        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):


    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:


        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def transform(self, results: dict) -> dict:


        img = results['img']

        img = self.brightness(img)


        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)


        img = self.saturation(img)


        img = self.hue(img)


        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

@TRANSFORMS.register_module()
class CustomizePhotoMetricDistortionGPT(BaseTransform):


    def __init__(self,
                 num=2,
                 brightness_delta: int = 64,
                 contrast_range: Sequence[float] = (0.3, 2.0),
                 saturation_range: Sequence[float] = (0.3, 2.0),
                 hue_delta: int = 36,
                 noise_std: float = 25.0,
                 aug_prob: float = 0.5,
                 seed: Optional[int] = None):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.noise_std = noise_std
        self.aug_prob = aug_prob


        self.augmentations = [
            RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.2, 0.2), p=1.0),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            GaussianBlur(blur_limit=(3, 7), p=1.0)
        ]

        if seed is not None:
            rd.seed(seed)
            nprd.seed(seed)

    def convert(self, img: np.ndarray, alpha: float = 1, beta: float = 0) -> np.ndarray:

        img = img.astype(np.float32) * alpha + beta
        return np.clip(img, 0, 255).astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:

        if nprd.random() < self.aug_prob:
            beta = nprd.uniform(-self.brightness_delta, self.brightness_delta)
            img = self.convert(img, beta=beta)
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:

        if nprd.random() < self.aug_prob:
            alpha = nprd.uniform(self.contrast_lower, self.contrast_upper)
            img = self.convert(img, alpha=alpha)
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:

        img_hsv = mmcv.bgr2hsv(img)
        if nprd.random() < self.aug_prob:
            img_hsv[:, :, 1] = self.convert(
                img_hsv[:, :, 1], alpha=nprd.uniform(self.saturation_lower, self.saturation_upper)
            )
        img = mmcv.hsv2bgr(img_hsv)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:

        img_hsv = mmcv.bgr2hsv(img)
        if nprd.random() < self.aug_prob:
            img_hsv[:, :, 0] = (img_hsv[:, :, 0].astype(int) +
                                nprd.randint(-self.hue_delta, self.hue_delta)) % 180
        img = mmcv.hsv2bgr(img_hsv)
        return img

    def add_noise(self, img: np.ndarray) -> np.ndarray:

        if nprd.random() < self.aug_prob:
            noise = nprd.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def apply_augmentations(self, img: np.ndarray) -> np.ndarray:

        num_augmentations = nprd.randint(1, len(self.augmentations))
        chosen_augmentations = sample(self.augmentations, num_augmentations)
        augmentations = Compose(chosen_augmentations)
        img = augmentations(image=img)['image']
        return img

    def apply_transform(self, img: np.ndarray) -> np.ndarray:


        img = self.brightness(img)
        mode = nprd.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)


        img = self.add_noise(img)


        if nprd.random() < self.aug_prob:
            img = self.apply_augmentations(img)

        return img

    def transform(self, results: dict) -> dict:

        img = results['img']


        img_original = img[:, :, 0:3]
        img_mixup = img[:, :, 3:6]


        img_original = self.apply_transform(img_original)
        img_mixup = self.apply_transform(img_mixup)


        results['img'] = np.concatenate([img_original, img_mixup], axis=2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, {self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, {self.saturation_upper}), '
                     f'hue_delta={self.hue_delta}, '
                     f'noise_std={self.noise_std}, '
                     f'aug_prob={self.aug_prob}, '
                     f'seed={self.seed})')
        return repr_str

@TRANSFORMS.register_module()
class PhotoMetricDistortionWithStyle(BaseTransform):


    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18,
                 grayscale_prob: float = 1.0):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.grayscale_prob = grayscale_prob

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:


        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:


        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img


    def transform(self, results: dict) -> dict:


        img = results['img']
        img_style = results['img_style']


        img = self.brightness(img)


        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)


        img = self.saturation(img)


        img = self.hue(img)


        if mode == 0:
            img = self.contrast(img)


        if img.shape!= img_style.shape:

            img_style_resized = Image.fromarray(img_style.astype(np.uint8))
            img_style_resized = img_style_resized.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
            img_style = np.array(img_style_resized).astype(np.float32)


        img_combined = np.concatenate([img, img_style], axis=2)


        results['img'] = img_combined
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@TRANSFORMS.register_module()
class ConcatStyle(BaseTransform):
    def transform(self, results: dict) -> dict:
        img = results['img']
        img_style = results['img_style']

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = np.stack([gray, gray, gray], axis=2)


        if img.shape!= img_style.shape:

            img_style_resized = Image.fromarray(img_style.astype(np.uint8))
            img_style_resized = img_style_resized.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
            img_style = np.array(img_style_resized).astype(np.float32)


        img_combined = np.concatenate([img, img_style], axis=2)


        results['img'] = img_combined
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class PhotoMetricDistortionWithStyleStrong(BaseTransform):


    def __init__(self,
                 brightness_delta: int = 64,
                 contrast_range: Sequence[float] = (0.3, 2.0),
                 saturation_range: Sequence[float] = (0.3, 2.0),
                 hue_delta: int = 36,
                 noise_std: float = 25.0,
                 aug_prob: float = 0.5,
                 seed: Optional[int] = None):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.noise_std = noise_std
        self.aug_prob = aug_prob

        self.augmentations = [
            RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.2, 0.2), p=1.0),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            GaussianBlur(blur_limit=(3, 7), p=1.0)
        ]

        if seed is not None:
            random.seed(seed)

    def convert(self, img: np.ndarray, alpha: float = 1, beta: float = 0) -> np.ndarray:

        img = img.astype(np.float32) * alpha + beta
        return np.clip(img, 0, 255).astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:

        if random.random() < self.aug_prob:
            beta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = self.convert(img, beta=beta)
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:

        if random.random() < self.aug_prob:
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img = self.convert(img, alpha=alpha)
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:

        img_hsv = mmcv.bgr2hsv(img)
        if random.random() < self.aug_prob:
            img_hsv[:, :, 1] = self.convert(
                img_hsv[:, :, 1], alpha=random.uniform(self.saturation_lower, self.saturation_upper)
            )
        img = mmcv.hsv2bgr(img_hsv)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:

        img_hsv = mmcv.bgr2hsv(img)
        if random.random() < self.aug_prob:
            img_hsv[:, :, 0] = (img_hsv[:, :, 0].astype(int) +
                                random.randint(-self.hue_delta, self.hue_delta)) % 180
        img = mmcv.hsv2bgr(img_hsv)
        return img

    def add_noise(self, img: np.ndarray) -> np.ndarray:

        if random.random() < self.aug_prob:
            noise = random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def apply_augmentations(self, img: np.ndarray) -> np.ndarray:

        num_augmentations = random.randint(1, len(self.augmentations))
        chosen_augmentations = sample(self.augmentations,num_augmentations)
        augmentations = Compose(chosen_augmentations)
        img = augmentations(image=img)['image']
        return img

    def apply_transform(self, img: np.ndarray) -> np.ndarray:


        img = self.brightness(img)
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)


        img = self.add_noise(img)


        if random.random() < self.aug_prob:
            img = self.apply_augmentations(img)

        return img

    def transform(self, results: dict) -> dict:


        img = results['img']
        img_style = results['img_style']

        img = self.apply_transform(img)


        if img.shape!= img_style.shape:

            img_style_resized = Image.fromarray(img_style.astype(np.uint8))
            img_style_resized = img_style_resized.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
            img_style = np.array(img_style_resized).astype(np.float32)


        img_combined = np.concatenate([img, img_style], axis=2)


        results['img'] = img_combined
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

@TRANSFORMS.register_module()
class RandomCutOut(BaseTransform):


    def __init__(self,
                 prob,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0),
                 seg_fill_in=None):

        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_fill_in is not None:
            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
                    and seg_fill_in <= 255)
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    @cache_randomness
    def do_cutout(self):
        return np.random.rand() < self.prob

    @cache_randomness
    def generate_patches(self, results):
        cutout = self.do_cutout()

        h, w, _ = results['img'].shape
        if cutout:
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        else:
            n_holes = 0
        x1_lst = []
        y1_lst = []
        index_lst = []
        for _ in range(n_holes):
            x1_lst.append(np.random.randint(0, w))
            y1_lst.append(np.random.randint(0, h))
            index_lst.append(np.random.randint(0, len(self.candidates)))
        return cutout, n_holes, x1_lst, y1_lst, index_lst

    def transform(self, results: dict) -> dict:

        cutout, n_holes, x1_lst, y1_lst, index_lst = self.generate_patches(
            results)
        if cutout:
            h, w, c = results['img'].shape
            for i in range(n_holes):
                x1 = x1_lst[i]
                y1 = y1_lst[i]
                index = index_lst[i]
                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                results['img'][y1:y2, x1:x2, :] = self.fill_in

                if self.seg_fill_in is not None:
                    for key in results.get('seg_fields', []):
                        results[key][y1:y2, x1:x2] = self.seg_fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in}, '
        repr_str += f'seg_fill_in={self.seg_fill_in})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotFlip(BaseTransform):


    def __init__(self, rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        assert 0 <= rotate_prob <= 1 and 0 <= flip_prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'

    def random_rot_flip(self, results: dict) -> dict:
        k = np.random.randint(0, 4)
        results['img'] = np.rot90(results['img'], k)
        for key in results.get('seg_fields', []):
            results[key] = np.rot90(results[key], k)
        axis = np.random.randint(0, 2)
        results['img'] = np.flip(results['img'], axis=axis).copy()
        for key in results.get('seg_fields', []):
            results[key] = np.flip(results[key], axis=axis).copy()
        return results

    def random_rotate(self, results: dict) -> dict:
        angle = np.random.uniform(min(*self.degree), max(*self.degree))
        results['img'] = mmcv.imrotate(results['img'], angle=angle)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(results[key], angle=angle)
        return results

    def transform(self, results: dict) -> dict:


        rotate_flag = 0
        if random.random() < self.rotate_prob:
            results = self.random_rotate(results)
            rotate_flag = 1
        if random.random() < self.flip_prob and rotate_flag == 0:
            results = self.random_rot_flip(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_prob={self.rotate_prob}, ' \
                    f'flip_prob={self.flip_prob}, ' \
                    f'degree={self.degree})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(MMCV_RandomFlip):


    def _flip(self, results: dict) -> None:


        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]


        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])


        for key in results.get('seg_fields', []):
            if results.get(key, None) is not None:
                results[key] = self._flip_seg_map(
                    results[key], direction=results['flip_direction']).copy()
                results['swap_seg_labels'] = self.swap_seg_labels


@TRANSFORMS.register_module()
class Resize(MMCV_Resize):


    def _resize_seg(self, results: dict) -> None:

        for seg_key in results.get('seg_fields', []):
            if results.get(seg_key, None) is not None:
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                results[seg_key] = gt_seg

@TRANSFORMS.register_module()
class StyleResize(BaseTransform):


    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:


        if results.get('img_style', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img_style'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)


                new_h, new_w = img.shape[:2]
                h, w = results['img_style'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img_style'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img_style'] = img


    def transform(self, results: dict) -> dict:


        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img_style'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomMosaic(BaseTransform):


    def __init__(self,
                 prob,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 pad_val=0,
                 seg_pad_val=255):
        assert 0 <= prob and prob <= 1
        assert isinstance(img_scale, tuple)
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    @cache_randomness
    def do_mosaic(self):
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:


        mosaic = self.do_mosaic()
        if mosaic:
            results = self._mosaic_transform_img(results)
            results = self._mosaic_transform_seg(results)
        return results

    def get_indices(self, dataset: MultiImageMixDataset) -> list:


        indices = [random.randint(0, len(dataset)) for _ in range(3)]
        return indices

    @cache_randomness
    def generate_mosaic_center(self):

        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        return center_x, center_y

    def _mosaic_transform_img(self, results: dict) -> dict:


        assert 'mix_results' in results
        if len(results['img'].shape) == 3:
            c = results['img'].shape[2]
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), c),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)


        self.center_x, self.center_y = self.generate_mosaic_center()
        center_position = (self.center_x, self.center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                result_patch = copy.deepcopy(results)
            else:
                result_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = result_patch['img']
            h_i, w_i = img_i.shape[:2]

            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))


            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord


            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['ori_shape'] = mosaic_img.shape

        return results

    def _mosaic_transform_seg(self, results: dict) -> dict:


        assert 'mix_results' in results
        for key in results.get('seg_fields', []):
            mosaic_seg = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.seg_pad_val,
                dtype=results[key].dtype)


            center_position = (self.center_x, self.center_y)

            loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
            for i, loc in enumerate(loc_strs):
                if loc == 'top_left':
                    result_patch = copy.deepcopy(results)
                else:
                    result_patch = copy.deepcopy(results['mix_results'][i - 1])

                gt_seg_i = result_patch[key]
                h_i, w_i = gt_seg_i.shape[:2]

                scale_ratio_i = min(self.img_scale[0] / h_i,
                                    self.img_scale[1] / w_i)
                gt_seg_i = mmcv.imresize(
                    gt_seg_i,
                    (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
                    interpolation='nearest')


                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, gt_seg_i.shape[:2][::-1])
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord


                mosaic_seg[y1_p:y2_p, x1_p:x2_p] = \
                    gt_seg_i[y1_c:y2_c, x1_c:x2_c]

            results[key] = mosaic_seg

        return results

    def _mosaic_combine(self, loc: str, center_position_xy: Sequence[float],
                        img_shape_wh: Sequence[int]) -> tuple:


        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':

            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':

            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':

            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:

            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'seg_pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class GenerateEdge(BaseTransform):


    def __init__(self, edge_width: int = 3, ignore_index: int = 255) -> None:
        super().__init__()
        self.edge_width = edge_width
        self.ignore_index = ignore_index

    def transform(self, results: Dict) -> Dict:


        h, w = results['img_shape']
        edge = np.zeros((h, w), dtype=np.uint8)
        seg_map = results['gt_seg_map']


        edge_down = edge[1:h, :]
        edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
                  & (seg_map[1:h, :] != self.ignore_index) &
                  (seg_map[:h - 1, :] != self.ignore_index)] = 1

        edge_left = edge[:, :w - 1]
        edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
                  & (seg_map[:, :w - 1] != self.ignore_index) &
                  (seg_map[:, 1:w] != self.ignore_index)] = 1

        edge_upleft = edge[:h - 1, :w - 1]
        edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                    & (seg_map[:h - 1, :w - 1] != self.ignore_index) &
                    (seg_map[1:h, 1:w] != self.ignore_index)] = 1

        edge_upright = edge[:h - 1, 1:w]
        edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                     & (seg_map[:h - 1, 1:w] != self.ignore_index) &
                     (seg_map[1:h, :w - 1] != self.ignore_index)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.edge_width, self.edge_width))
        edge = cv2.dilate(edge, kernel)

        results['gt_edge_map'] = edge
        results['edge_width'] = self.edge_width

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'edge_width={self.edge_width}, '
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str


@TRANSFORMS.register_module()
class ResizeShortestEdge(BaseTransform):


    def __init__(self, scale: Union[int, Tuple[int, int]],
                 max_size: int) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size


        self.resize = TRANSFORMS.build({
            'type': 'Resize',
            'scale': 0,
            'keep_ratio': True
        })

    def _get_output_shape(self, img, short_edge_length) -> Tuple[int, int]:


        h, w = img.shape[:2]
        if isinstance(short_edge_length, int):
            size = short_edge_length * 1.0
        elif isinstance(short_edge_length, tuple):
            size = min(short_edge_length) * 1.0
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h *= scale
            new_w *= scale

        new_h = int(new_h + 0.5)
        new_w = int(new_w + 0.5)
        return (new_w, new_h)

    def transform(self, results: Dict) -> Dict:
        self.resize.scale = self._get_output_shape(results['img'], self.scale)
        return self.resize(results)


@TRANSFORMS.register_module()
class BioMedical3DRandomCrop(BaseTransform):


    def __init__(self,
                 crop_shape: Union[int, Tuple[int, int, int]],
                 keep_foreground: bool = True):
        super().__init__()
        assert isinstance(crop_shape, int) or (
            isinstance(crop_shape, tuple) and len(crop_shape) == 3
        ), 'The expected crop_shape is an integer, or a tuple containing '


        if isinstance(crop_shape, int):
            crop_shape = (crop_shape, crop_shape, crop_shape)
        assert crop_shape[0] > 0 and crop_shape[1] > 0 and crop_shape[2] > 0
        self.crop_shape = crop_shape
        self.keep_foreground = keep_foreground

    def random_sample_location(self, seg_map: np.ndarray) -> dict:


        num_samples = 10000


        min_percent_coverage = 0.01
        class_locs = {}
        foreground_classes = []
        all_classes = np.unique(seg_map)
        for c in all_classes:
            if c == 0:


                class_locs[c] = []
            else:
                all_locs = np.argwhere(seg_map == c)
                target_num_samples = min(num_samples, len(all_locs))
                target_num_samples = max(
                    target_num_samples,
                    int(np.ceil(len(all_locs) * min_percent_coverage)))

                selected = all_locs[np.random.choice(
                    len(all_locs), target_num_samples, replace=False)]
                class_locs[c] = selected
                foreground_classes.append(c)

        selected_voxel = None
        if len(foreground_classes) > 0:
            selected_class = np.random.choice(foreground_classes)
            voxels_of_that_class = class_locs[selected_class]
            selected_voxel = voxels_of_that_class[np.random.choice(
                len(voxels_of_that_class))]

        return selected_voxel

    def random_generate_crop_bbox(self, margin_z: int, margin_y: int,
                                  margin_x: int) -> tuple:


        offset_z = np.random.randint(0, margin_z + 1)
        offset_y = np.random.randint(0, margin_y + 1)
        offset_x = np.random.randint(0, margin_x + 1)
        crop_z1, crop_z2 = offset_z, offset_z + self.crop_shape[0]
        crop_y1, crop_y2 = offset_y, offset_y + self.crop_shape[1]
        crop_x1, crop_x2 = offset_x, offset_x + self.crop_shape[2]

        return crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2

    def generate_margin(self, results: dict) -> tuple:


        seg_map = results['gt_seg_map']
        if self.keep_foreground:
            selected_voxel = self.random_sample_location(seg_map)
            if selected_voxel is None:


                warnings.warn(f'case does not contain any foreground classes'
                              f': {results["img_path"]}')
                margin_z = max(seg_map.shape[0] - self.crop_shape[0], 0)
                margin_y = max(seg_map.shape[1] - self.crop_shape[1], 0)
                margin_x = max(seg_map.shape[2] - self.crop_shape[2], 0)
            else:
                margin_z = max(0, selected_voxel[0] - self.crop_shape[0] // 2)
                margin_y = max(0, selected_voxel[1] - self.crop_shape[1] // 2)
                margin_x = max(0, selected_voxel[2] - self.crop_shape[2] // 2)
                margin_z = max(
                    0, min(seg_map.shape[0] - self.crop_shape[0], margin_z))
                margin_y = max(
                    0, min(seg_map.shape[1] - self.crop_shape[1], margin_y))
                margin_x = max(
                    0, min(seg_map.shape[2] - self.crop_shape[2], margin_x))
        else:
            margin_z = max(seg_map.shape[0] - self.crop_shape[0], 0)
            margin_y = max(seg_map.shape[1] - self.crop_shape[1], 0)
            margin_x = max(seg_map.shape[2] - self.crop_shape[2], 0)

        return margin_z, margin_y, margin_x

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:


        crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if len(img.shape) == 3:

            img = img[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        else:

            assert len(img.shape) == 4
            img = img[:, crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def transform(self, results: dict) -> dict:


        margin = self.generate_margin(results)
        crop_bbox = self.random_generate_crop_bbox(*margin)


        img = results['img']
        results['img'] = self.crop(img, crop_bbox)
        results['img_shape'] = results['img'].shape[1:]


        seg_map = results['gt_seg_map']
        results['gt_seg_map'] = self.crop(seg_map, crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_shape={self.crop_shape})'


@TRANSFORMS.register_module()
class BioMedicalGaussianNoise(BaseTransform):


    def __init__(self,
                 prob: float = 0.1,
                 mean: float = 0.0,
                 std: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= prob <= 1.0 and std >= 0.0
        self.prob = prob
        self.mean = mean
        self.std = std

    def transform(self, results: Dict) -> Dict:


        if np.random.rand() < self.prob:
            rand_std = np.random.uniform(0, self.std)
            noise = np.random.normal(
                self.mean, rand_std, size=results['img'].shape)

            noise = noise.astype(results['img'].dtype)
            results['img'] = results['img'] + noise
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'mean={self.mean}, '
        repr_str += f'std={self.std})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedicalGaussianBlur(BaseTransform):


    def __init__(self,
                 sigma_range: Tuple[float, float] = (0.5, 1.0),
                 prob: float = 0.2,
                 prob_per_channel: float = 0.5,
                 different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = True) -> None:
        super().__init__()
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= prob_per_channel <= 1.0
        assert isinstance(sigma_range, Sequence) and len(sigma_range) == 2
        self.sigma_range = sigma_range
        self.prob = prob
        self.prob_per_channel = prob_per_channel
        self.different_sigma_per_channel = different_sigma_per_channel
        self.different_sigma_per_axis = different_sigma_per_axis

    def _get_valid_sigma(self, value_range) -> Tuple[float, ...]:


        if (isinstance(value_range, (list, tuple))):
            if (value_range[0] == value_range[1]):
                value = value_range[0]
            else:
                orig_type = type(value_range[0])
                value = np.random.uniform(value_range[0], value_range[1])
                value = orig_type(value)
        return value

    def _gaussian_blur(self, data_sample: np.ndarray) -> np.ndarray:


        sigma = None
        for c in range(data_sample.shape[0]):
            if np.random.rand() < self.prob_per_channel:


                if (sigma is None or self.different_sigma_per_channel):
                    if (not self.different_sigma_per_axis):
                        sigma = self._get_valid_sigma(self.sigma_range)
                    else:
                        sigma = [
                            self._get_valid_sigma(self.sigma_range)
                            for _ in data_sample.shape[1:]
                        ]

                data_sample[c] = gaussian_filter(
                    data_sample[c], sigma, order=0)
        return data_sample

    def transform(self, results: Dict) -> Dict:


        if np.random.rand() < self.prob:
            results['img'] = self._gaussian_blur(results['img'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'prob_per_channel={self.prob_per_channel}, '
        repr_str += f'sigma_range={self.sigma_range}, '
        repr_str += 'different_sigma_per_channel=' \
                    f'{self.different_sigma_per_channel}, '
        repr_str += 'different_sigma_per_axis=' \
                    f'{self.different_sigma_per_axis})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedicalRandomGamma(BaseTransform):


    def __init__(self,
                 prob: float = 0.5,
                 gamma_range: Tuple[float] = (0.5, 2),
                 invert_image: bool = False,
                 per_channel: bool = False,
                 retain_stats: bool = False):
        assert 0 <= prob and prob <= 1
        assert isinstance(gamma_range, tuple) and len(gamma_range) == 2
        assert isinstance(invert_image, bool)
        assert isinstance(per_channel, bool)
        assert isinstance(retain_stats, bool)
        self.prob = prob
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    @cache_randomness
    def _do_gamma(self):

        return np.random.rand() < self.prob

    def _adjust_gamma(self, img: np.array):


        if self.invert_image:
            img = -img

        def _do_adjust(img):
            if retain_stats_here:
                img_mean = img.mean()
                img_std = img.std()
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(
                    max(self.gamma_range[0], 1), self.gamma_range[1])
            img_min = img.min()
            img_range = img.max() - img_min
            img = np.power(((img - img_min) / float(img_range + 1e-7)),
                           gamma) * img_range + img_min
            if retain_stats_here:
                img = img - img.mean()
                img = img / (img.std() + 1e-8) * img_std
                img = img + img_mean
            return img

        if not self.per_channel:
            retain_stats_here = self.retain_stats
            img = _do_adjust(img)
        else:
            for c in range(img.shape[0]):
                img[c] = _do_adjust(img[c])
        if self.invert_image:
            img = -img
        return img

    def transform(self, results: dict) -> dict:


        do_gamma = self._do_gamma()

        if do_gamma:
            results['img'] = self._adjust_gamma(results['img'])
        else:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'gamma_range={self.gamma_range},'
        repr_str += f'invert_image={self.invert_image},'
        repr_str += f'per_channel={self.per_channel},'
        repr_str += f'retain_stats={self.retain_stats}'
        return repr_str


@TRANSFORMS.register_module()
class BioMedical3DPad(BaseTransform):


    def __init__(self,
                 pad_shape: Tuple[int, int, int],
                 pad_val: float = 0.,
                 seg_pad_val: int = 0) -> None:


        assert pad_shape is not None
        if not isinstance(pad_shape, tuple):
            assert len(pad_shape) == 3

        self.pad_shape = pad_shape
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, results: dict) -> None:


        padded_img = self._to_pad(
            results['img'], pad_shape=self.pad_shape, pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape[1:]

    def _pad_seg(self, results: dict) -> None:


        if results.get('gt_seg_map', None) is not None:
            pad_gt_seg = self._to_pad(
                results['gt_seg_map'][None, ...],
                pad_shape=results['pad_shape'],
                pad_val=self.seg_pad_val)
            results['gt_seg_map'] = pad_gt_seg[1:]

    @staticmethod
    def _to_pad(img: np.ndarray,
                pad_shape: Tuple[int, int, int],
                pad_val: Union[int, float] = 0) -> np.ndarray:


        d = max(pad_shape[0] - img.shape[1], 0)
        pad_d = (d // 2, d - d // 2)
        h = max(pad_shape[1] - img.shape[2], 0)
        pad_h = (h // 2, h - h // 2)
        w = max(pad_shape[2] - img.shape[2], 0)
        pad_w = (w // 2, w - w // 2)

        pad_list = [(0, 0), pad_d, pad_h, pad_w]

        img = np.pad(img, pad_list, mode='constant', constant_values=pad_val)
        return img

    def transform(self, results: dict) -> dict:


        self._pad_img(results)
        self._pad_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'pad_shape={self.pad_shape}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'seg_pad_val={self.seg_pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedical3DRandomFlip(BaseTransform):


    def __init__(self,
                 prob: float,
                 axes: Tuple[int, ...],
                 swap_label_pairs: Optional[List[Tuple[int, int]]] = None):
        self.prob = prob
        self.axes = axes
        self.swap_label_pairs = swap_label_pairs
        assert prob >= 0 and prob <= 1
        if axes is not None:
            assert max(axes) <= 2

    @staticmethod
    def _flip(img, direction: Tuple[bool, bool, bool]) -> np.ndarray:
        if direction[0]:
            img[:, :] = img[:, ::-1]
        if direction[1]:
            img[:, :, :] = img[:, :, ::-1]
        if direction[2]:
            img[:, :, :, :] = img[:, :, :, ::-1]
        return img

    def _do_flip(self, img: np.ndarray) -> Tuple[bool, bool, bool]:


        flip_c, flip_x, flip_y = False, False, False
        if self.axes is not None:
            flip_c = 0 in self.axes and np.random.rand() < self.prob
            flip_x = 1 in self.axes and np.random.rand() < self.prob
            if len(img.shape) == 4:
                flip_y = 2 in self.axes and np.random.rand() < self.prob
        return flip_c, flip_x, flip_y

    def _swap_label(self, seg: np.ndarray) -> np.ndarray:
        out = seg.copy()
        for first, second in self.swap_label_pairs:
            first_area = (seg == first)
            second_area = (seg == second)
            out[first_area] = second
            out[second_area] = first
        return out

    def transform(self, results: Dict) -> Dict:


        if 'do_flip' not in results:
            results['do_flip'] = self._do_flip(results['img'])
        if 'flip_axes' not in results:
            results['flip_axes'] = self.axes

        results['img'] = self._flip(
            results['img'], direction=results['do_flip'])

        if results['gt_seg_map'] is not None:
            if results['gt_seg_map'].shape != results['img'].shape:
                results['gt_seg_map'] = results['gt_seg_map'][None, :]
            results['gt_seg_map'] = self._flip(
                results['gt_seg_map'], direction=results['do_flip'])
            results['gt_seg_map'] = results['gt_seg_map'].squeeze()

            if self.swap_label_pairs is not None:
                results['gt_seg_map'] = self._swap_label(results['gt_seg_map'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, axes={self.axes}, ' \
                    f'swap_label_pairs={self.swap_label_pairs})'
        return repr_str


@TRANSFORMS.register_module()
class Albu(BaseTransform):


    def __init__(self,
                 transforms: List[dict],
                 keymap: Optional[dict] = None,
                 update_pad_shape: bool = False):
        if not ALBU_INSTALLED:
            raise ImportError(
                'albumentations is not installed, '
                'we suggest install albumentation by '
                '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'
            )


        transforms = copy.deepcopy(transforms)

        self.transforms = transforms
        self.keymap = keymap
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {'img': 'image', 'gt_seg_map': 'mask'}
        else:
            self.keymap_to_albu = copy.deepcopy(keymap)
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> object:


        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            if not ALBU_INSTALLED:
                raise ImportError(
                    'albumentations is not installed, '
                    'we suggest install albumentation by '
                    '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'
                )
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a valid type or str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(t) for t in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict):


        updated_dict = {}
        for k, _ in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def transform(self, results):

        results = self.mapper(results, self.keymap_to_albu)


        results['image'] = cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB)

        results = self.aug(**results)


        results['image'] = cv2.cvtColor(results['image'], cv2.COLOR_RGB2BGR)


        results = self.mapper(results, self.keymap_back)


        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@TRANSFORMS.register_module()
class ConcatCDInput(BaseTransform):


    def __init__(self, input_keys=('img', 'img2')):
        self.input_keys = input_keys

    def transform(self, results: dict) -> dict:
        img = []
        for input_key in self.input_keys:
            img.append(results.pop(input_key))
        results['img'] = np.concatenate(img, axis=2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(input_keys={self.input_keys}, '
        return repr_str


@TRANSFORMS.register_module()
class RandomDepthMix(BaseTransform):


    def __init__(
        self,
        prob: float = 0.25,
        mix_scale_ratio: float = 0.75,
    ):
        super().__init__()

        self.prob = prob
        self.mix_scale_ratio = mix_scale_ratio

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        h, w = results['img_shape'][:2]
        left = int(w * random.random())
        width_ratio = self.mix_scale_ratio * random.random()
        width = int(max(1, (w - left) * width_ratio))

        img = results['img']
        depth_rescale_factor = results.get('depth_rescale_factor', 1)
        depth_map = results['gt_depth_map'] / depth_rescale_factor

        if img.ndim == 3:
            for c in range(img.shape[-1]):
                img[:, left:left + width, c] = depth_map[:, left:left + width]
        elif img.ndim == 2:
            img[:, left:left + width] = depth_map[:, left:left + width]
        else:
            raise ValueError(f'Invalid image shape ({img.shape})')

        results['img'] = img
        return results
