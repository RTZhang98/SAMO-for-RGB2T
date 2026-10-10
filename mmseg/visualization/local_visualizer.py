from typing import Dict, List, Optional

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer

from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.utils import get_classes, get_palette


@VISUALIZERS.register_module()
class SegLocalVisualizer(Visualizer):


    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs)
        self.alpha: float = alpha
        self.set_dataset_meta(palette, classes, dataset_name)

    def _get_center_loc(self, mask: np.ndarray) -> np.ndarray:


        loc = np.argwhere(mask == 1)

        loc_sort = np.array(
            sorted(loc.tolist(), key=lambda row: (row[0], row[1])))
        y_list = loc_sort[:, 0]
        unique, indices, counts = np.unique(
            y_list, return_index=True, return_counts=True)
        y_loc = unique[counts.argmax()]
        y_most_freq_loc = loc[loc_sort[:, 0] == y_loc]
        center_num = len(y_most_freq_loc) // 2
        x = y_most_freq_loc[center_num][1]
        y = y_most_freq_loc[center_num][0]
        return np.array([x, y])

    def _draw_sem_seg(self,
                      image: np.ndarray,
                      sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List],
                      with_labels: Optional[bool] = True) -> np.ndarray:


        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        if with_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX

            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            fontColor = (255, 255, 255)
            if image.shape[0] < 300 or image.shape[1] < 300:
                thickness = 1
                rectangleThickness = 1
            else:
                thickness = 2
                rectangleThickness = 2
            lineType = 2

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            masks = masks.astype(np.uint8)
            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(masks[mask_num])
                text = classes[classes_id]
                (label_width, label_height), baseline = cv2.getTextSize(
                    text, font, fontScale, thickness)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     classes_color, -1)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     (0, 0, 0), rectangleThickness)
                mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                                   font, fontScale, fontColor, thickness,
                                   lineType)
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg

    def _draw_depth_map(self, image: np.ndarray,
                        depth_map: PixelData) -> np.ndarray:


        depth_map = depth_map.cpu().data
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        if depth_map.ndim == 2:
            depth_map = depth_map[None]

        depth_map = self.draw_featmap(depth_map, resize_shape=image.shape[:2])
        out_image = np.concatenate((image, depth_map), axis=0)
        self.set_image(out_image)
        return out_image

    def set_dataset_meta(self,
                         classes: Optional[List] = None,
                         palette: Optional[List] = None,
                         dataset_name: Optional[str] = None) -> None:


        if dataset_name is None:
            dataset_name = 'cityscapes'
        classes = classes if classes else get_classes(dataset_name)
        palette = palette if palette else get_palette(dataset_name)
        assert len(classes) == len(
            palette), 'The length of classes should be equal to palette'
        self.dataset_meta: dict = {'classes': classes, 'palette': palette}

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,

            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True) -> None:


        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_sem_seg(image, data_sample.gt_sem_seg,
                                                 classes, palette, with_labels)

            if 'gt_depth_map' in data_sample:
                gt_img_data = gt_img_data if gt_img_data is not None else image
                gt_img_data = self._draw_depth_map(gt_img_data,
                                                   data_sample.gt_depth_map)

        if draw_pred and data_sample is not None:

            if 'pred_sem_seg' in data_sample:

                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_sem_seg(image,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette,
                                                   with_labels)

            if 'pred_depth_map' in data_sample:
                pred_img_data = pred_img_data if pred_img_data is not None \
                    else image
                pred_img_data = self._draw_depth_map(
                    pred_img_data, data_sample.pred_depth_map)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)
