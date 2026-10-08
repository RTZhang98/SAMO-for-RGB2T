import os.path as osp
from typing import Dict, Sequence

import numpy as np
from mmengine.logging import MMLogger, print_log
from PIL import Image

from mmseg.registry import METRICS
from collections import OrderedDict
from prettytable import PrettyTable
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from collections import defaultdict
import math

@METRICS.register_module()
class DGIoUMetric(IoUMetric):
    def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
        super().__init__(**kwargs)
        self.dataset_keys = dataset_keys
        if mean_used_keys:
            self.mean_used_keys = mean_used_keys
        else:
            self.mean_used_keys = dataset_keys

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()

            if not self.format_only:
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
                res1, res2, res3, res4 = self.intersect_and_union(
                    pred_label, label, num_classes, self.ignore_index
                )
                dataset_key = "unknown"
                for key in self.dataset_keys:
                    if key in data_samples[0]["seg_map_path"]:
                        dataset_key = key
                        break
                self.results.append([dataset_key, res1, res2, res3, res4])

            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(data_sample["img_path"]))[0]
                png_filename = osp.abspath(osp.join(self.output_dir, f"{basename}.png"))
                output_mask = pred_label.cpu().numpy()

                if data_sample.get("reduce_zero_label", False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            dataset_results[result[0]].append(result[1:])
        metrics_type2mean = defaultdict(list)
        for key, key_result in dataset_results.items():
            logger: MMLogger = MMLogger.get_current_instance()
            print_log(f"----------metrics for {key}------------", logger)
            key_metrics = super().compute_metrics(key_result)
            print_log(f"number of samples for {key}: {len(key_result)}")
            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v
                if key in self.mean_used_keys:
                    metrics_type2mean[k].append(v)
        for k, v in metrics_type2mean.items():
            metrics[f"mean_{k}"] = sum(v) / len(v)
        return metrics

@METRICS.register_module()
class DGIoUMetricThermal(IoUMetric):
    def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
        super().__init__(**kwargs)
        self.dataset_keys = dataset_keys
        if mean_used_keys:
            self.mean_used_keys = mean_used_keys
        else:
            self.mean_used_keys = dataset_keys

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()

            if not self.format_only:
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
                res1, res2, res3, res4 = self.intersect_and_union(
                    pred_label, label, num_classes, self.ignore_index
                )
                dataset_key = "unknown"
                for key in self.dataset_keys:
                    if key in data_samples[0]["seg_map_path"]:
                        dataset_key = key
                        break
                self.results.append([dataset_key, res1, res2, res3, res4])

            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(data_sample["img_path"]))[0]
                png_filename = osp.abspath(osp.join(self.output_dir, f"{basename}.png"))
                output_mask = pred_label.cpu().numpy()

                if data_sample.get("reduce_zero_label", False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics_origin(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        def safe_mean(arr):
            arr = np.array(arr, dtype=np.float32)
            arr = arr[~np.isnan(arr)]
            arr = arr[arr != 0]
            if len(arr) == 0:
                return np.nan
            return np.mean(arr)

        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(safe_mean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    def compute_metrics(self, results: list) -> Dict[str, float]:
        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            dataset_results[result[0]].append(result[1:])
        metrics_type2mean = defaultdict(list)
        for key, key_result in dataset_results.items():
            logger: MMLogger = MMLogger.get_current_instance()
            print_log(f"----------metrics for {key}------------", logger)
            key_metrics = self.compute_metrics_origin(key_result)
            print_log(f"number of samples for {key}: {len(key_result)}")
            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v
                if key in self.mean_used_keys:
                    metrics_type2mean[k].append(v)
        for k, v_list in metrics_type2mean.items():
            filtered_v = [
                float(v)
                for v in v_list
                if not (float(v) == 0 or math.isnan(float(v)))
            ]
            if filtered_v:
                metrics[f"mean_{k}"] = sum(filtered_v) / len(filtered_v)
            else:
                metrics[f"mean_{k}"] = float('nan')
        return metrics