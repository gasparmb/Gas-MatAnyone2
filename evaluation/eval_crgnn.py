# This file is modified based on `evaluate_hr.py` from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)
# Changes:
# - Adapted for CRGNN dataset
# - Supported metrics: pha_mad, pha_mse, pha_grad, pha_dtssd

"""
HR (High-Resolution) evaluation. We found using numpy is very slow for high resolution, so we moved it to PyTorch using CUDA.

Note, the script only does evaluation. You will need to first inference yourself and save the results to disk
Expected directory format for both prediction and ground-truth is:

    crgnn
        ├── video1
          ├── pha
            ├── 0000.png
        ├── video2
          ├── pha
            ├── 0000.png

Prediction must have the exact file structure and file name as the ground-truth,
except for the file extension.

Example usage:

python evaluation/eval_crgnn.py \
    --pred-dir ./data/results/crgnn \
    --true-dir ./data/crgnn/alpha 

An excel sheet with evaluation results will be written to "PATH_TO_PREDICTIONS/crgnn.xlsx"
"""


import argparse
import os
import cv2
import kornia
import numpy as np
import xlsxwriter
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class Evaluator:
    def __init__(self):
        self.parse_args()
        self.init_metrics()
        self.evaluate()
        self.write_excel()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--pred-dir', type=str, required=True)
        parser.add_argument('--true-dir', type=str, required=True)
        parser.add_argument('--num-workers', type=int, default=48)
        parser.add_argument('--metrics', type=str, nargs='+', default=[
            'pha_mad', 'pha_mse', 'pha_grad', 'pha_dtssd'])
        self.args = parser.parse_args()
        
    def init_metrics(self):
        self.mad = MetricMAD()
        self.mse = MetricMSE()
        self.grad = MetricGRAD()
        self.dtssd = MetricDTSSD()
        
    def evaluate(self):
        tasks = []
        position = 0
        
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            for video_folder in sorted(os.listdir(self.args.pred_dir)):
                pred_video_path = os.path.join(self.args.pred_dir, video_folder)
                true_video_path = os.path.join(self.args.true_dir, video_folder)
                if os.path.isdir(pred_video_path) and os.path.isdir(true_video_path):
                    future = executor.submit(self.evaluate_worker, video_folder, position)
                    tasks.append((video_folder, future))
                    position += 1
                    
        self.results = [(video_folder, future.result()) for video_folder, future in tasks]
        
    def write_excel(self):
        workbook = xlsxwriter.Workbook(os.path.join(self.args.pred_dir, f'{os.path.basename(self.args.pred_dir)}.xlsx'))
        summarysheet = workbook.add_worksheet('summary')
        metricsheets = [workbook.add_worksheet(metric) for metric in self.results[0][-1].keys()]
        
        for i, metric in enumerate(self.results[0][-1].keys()):
            summarysheet.write(i, 0, metric)
            summarysheet.write(i, 1, f'={metric}!B2')
        
        for row, (video_folder, metrics) in enumerate(self.results):
            for metricsheet, metric in zip(metricsheets, metrics.values()):
                # Write the header
                if row == 0:
                    metricsheet.write(1, 0, 'Average')
                    metricsheet.write(1, 1, f'=AVERAGE(C2:ZZ2)')
                    for col in range(len(metric)):
                        metricsheet.write(0, col + 2, col)
                        colname = xlsxwriter.utility.xl_col_to_name(col + 2)
                        metricsheet.write(1, col + 2, f'=AVERAGE({colname}3:{colname}9999)')
                        
                metricsheet.write(row + 2, 0, video_folder)
                metricsheet.write_row(row + 2, 1, metric)
        
        workbook.close()

    def evaluate_worker(self, video_folder, position):
        pred_pha_dir = os.path.join(self.args.pred_dir, video_folder, 'pha')
        true_pha_dir = os.path.join(self.args.true_dir, video_folder)
        
        # Get sorted filenames from both directories
        pred_framenames = sorted([f for f in os.listdir(pred_pha_dir) if os.path.isfile(os.path.join(pred_pha_dir, f))])
        true_framenames = sorted([f for f in os.listdir(true_pha_dir) if os.path.isfile(os.path.join(true_pha_dir, f))])
        
        # true-dir contains frames at every 10 frames (frame 1, 11, 21, ...)
        # pred-dir contains all frames (frame 0, 1, 2, 3, ...)
        # Match: true frame i (which is video frame i*10) corresponds to pred frame i*10
        num_true_frames = len(true_framenames)
        num_pred_frames = len(pred_framenames)
        
        if num_true_frames == 0:
            print(f'Warning: {video_folder} has no true frames')
            return {metric_name : [] for metric_name in self.args.metrics}
        
        print(f'Info: {video_folder} has {num_pred_frames} pred frames and {num_true_frames} true frames (every 10 frames)')
        print(f'  First few pred files: {pred_framenames[:5]}')
        print(f'  First few true files: {true_framenames[:5]}')
        
        metrics = {metric_name : [] for metric_name in self.args.metrics}
        
        pred_pha_tm1 = None
        true_pha_tm1 = None
        
        for true_idx in tqdm(range(num_true_frames), desc=video_folder, position=position, dynamic_ncols=True):
            # true frame at index true_idx corresponds to video frame true_idx * 10
            # pred frame at index pred_idx = true_idx * 10
            pred_idx = true_idx * 10
            
            if pred_idx >= num_pred_frames:
                print(f'Warning: {video_folder} true frame {true_idx} (video frame {pred_idx}) exceeds pred frames ({num_pred_frames})')
                break
            
            pred_pha_path = os.path.join(pred_pha_dir, pred_framenames[pred_idx])
            true_pha_path = os.path.join(true_pha_dir, true_framenames[true_idx])
            
            # Print the matched pair
            print(f'  [{video_folder}] Pair {true_idx}: pred[{pred_idx}]="{pred_framenames[pred_idx]}" <-> true[{true_idx}]="{true_framenames[true_idx]}"')
            
            pred_pha = cv2.imread(pred_pha_path, cv2.IMREAD_GRAYSCALE)
            true_pha = cv2.imread(true_pha_path, cv2.IMREAD_GRAYSCALE)
            
            if pred_pha is None or true_pha is None:
                print(f'Warning: Failed to read image at true_idx {true_idx} (pred_idx {pred_idx}) in {video_folder}')
                continue
            
            true_pha = torch.from_numpy(true_pha).cuda(non_blocking=True).float().div_(255)
            pred_pha = torch.from_numpy(pred_pha).cuda(non_blocking=True).float().div_(255)
            
            if 'pha_mad' in self.args.metrics:
                metrics['pha_mad'].append(self.mad(pred_pha, true_pha))
            if 'pha_mse' in self.args.metrics:
                metrics['pha_mse'].append(self.mse(pred_pha, true_pha))
            if 'pha_grad' in self.args.metrics:
                metrics['pha_grad'].append(self.grad(pred_pha, true_pha))
            if 'pha_conn' in self.args.metrics:
                metrics['pha_conn'].append(self.conn(pred_pha, true_pha))
            if 'pha_dtssd' in self.args.metrics:
                if true_idx == 0:
                    metrics['pha_dtssd'].append(0)
                else:
                    metrics['pha_dtssd'].append(self.dtssd(pred_pha, pred_pha_tm1, true_pha, true_pha_tm1))
                    
            pred_pha_tm1 = pred_pha
            true_pha_tm1 = true_pha
            

        return metrics

    
class MetricMAD:
    def __call__(self, pred, true):
        return (pred - true).abs_().mean() * 1e3


class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.filter_x = torch.from_numpy(self.filter_x).unsqueeze(0).cuda()
        self.filter_y = torch.from_numpy(self.filter_y).unsqueeze(0).cuda()
    
    def __call__(self, pred, true):
        true_grad = self.gauss_gradient(true)
        pred_grad = self.gauss_gradient(pred)
        return ((true_grad - pred_grad) ** 2).sum() / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = kornia.filters.filter2d(img[None, None, :, :], self.filter_x, border_type='replicate')[0, 0]
        img_filtered_y = kornia.filters.filter2d(img[None, None, :, :], self.filter_y, border_type='replicate')[0, 0]
        return (img_filtered_x**2 + img_filtered_y**2).sqrt()
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y
        
    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2


class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = dtSSD.sum() / true_t.numel()
        dtSSD = dtSSD.sqrt()
        return dtSSD * 1e2


if __name__ == '__main__':
    Evaluator()