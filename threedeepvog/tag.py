


#%% script_05_dv3d_threaded_classes.py
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os
# import sys    # sys.path.append("D:/git/DeepVOG3DTorch/DeepVOG/deepvog3D")
import torch
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from monai.transforms import Resize
import skvideo.io as skv
import kornia.enhance as kornia_enhance
import threading
# import plotly.offline as pyo
from concurrent.futures import ThreadPoolExecutor
# import skvideo.io as skv
# from astropy.convolution import convolve as nan_convolve
# from scipy.spatial import ConvexHull
# from deepvog3D.draw_ellipse_batch import fit_ellipse_compact

class EllipseFitting(threading.Thread):
    def __init__(self, threads, args, daemon=False, use_queue=True, device ='cpu'):
        super().__init__(daemon=daemon)
        self.name = 'Thread-EllipseFitting'
        self.args = args
        self.threads = threads
        self.use_queue = args.get('is_parallel', use_queue)  #priority to args['is_parallel'] if it exists
        self.device = args.get('device', device)
        self.blink_threshold = args.get('blink_threshold', 0.735)
        self.frame_counter = 0
        self.elapsed_time = 0
        H, W = args["vid_h"], args["vid_w"]  # or from seg shape later
        yy, xx = torch.meshgrid(torch.arange(H, device=self.device),
                                torch.arange(W, device=self.device), indexing="ij")
        self._xx = xx
        self._yy = yy

        # self.skimg_EllipseModel = measure.EllipseModel()
    
    @staticmethod
    def bwperim_batch(bw, n=4, mask=None):
        """
        perim = bwperim_torch(bw, n=4, mask=None)
        Find the perimeter of objects in binary images using PyTorch.
        A pixel is part of an object perimeter if its value is one and there
        is at least one zero-valued pixel in its neighborhood.
        By default the neighborhood of a pixel is 4 nearest pixels, but
        if `n` is set to 8 the 8 nearest pixels will be considered.
        
        Additionally, masks certain regions and boundaries if specified.
        
        Parameters
        ----------
        bw : A binary image tensor of shape (batch_size, height, width)
        n : Connectivity. Must be 4 or 8 (default: 4)
        mask : Optional mask tensor of the same shape as bw to exclude regions
        Returns
        -------
        perim : A boolean tensor of the same shape as bw
        """
        if n not in (4, 8):
            raise ValueError('bwperim_torch: n must be 4 or 8')
        # device = bw.device
        # batch_size, height, width = bw.shape
        # Pad the image with zeros on all sides
        padded_bw = torch.nn.functional.pad(bw, (1, 1, 1, 1), mode='constant', value=0)
        # Shifting operations
        north = padded_bw[:, :-2, 1:-1]
        south = padded_bw[:, 2:, 1:-1]
        west = padded_bw[:, 1:-1, :-2]
        east = padded_bw[:, 1:-1, 2:]
        
        # Initialize idx with 4-connectivity check
        idx = (north == bw) & (south == bw) & (west == bw) & (east == bw)
        if n == 8:
            north_east = padded_bw[:, :-2, 2:]
            north_west = padded_bw[:, :-2, :-2]
            south_east = padded_bw[:, 2:, 2:]
            south_west = padded_bw[:, 2:, :-2]
            idx &= (north_east == bw) & (north_west == bw) & (south_east == bw) & (south_west == bw)
        # The perimeter is the inverse of idx and masked by the original image
        perim = (~idx) * bw
        # masking bwperim_output on the img boundaries as 0 
        perim[:, 0, :] = False
        perim[:, -1, :] = False
        perim[:, :, 0] = False
        perim[:, :, -1] = False
        return perim
    
    @staticmethod
    def gen_ellipse_batch_info(perim: torch.Tensor, device):
        perim_np = perim.detach().cpu().numpy().astype(np.uint8)  # (B,H,W)
        B = perim_np.shape[0]

        centers = np.full((B, 2), np.nan, np.float32)
        ws = np.full((B,), np.nan, np.float32)
        hs = np.full((B,), np.nan, np.float32)
        radians = np.full((B,), np.nan, np.float32)
        valids = np.zeros((B,), dtype=bool)

        for i in range(B):
            cnts, _ = cv2.findContours(perim_np[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) < 6:
                continue

            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            # your convention: center=[y,x] -> but you store [x,y]
            centers[i] = [cx, cy]
            ws[i] = MA / 2.0
            hs[i] = ma / 2.0
            radians[i] = np.pi / 2 - np.deg2rad(angle)
            valids[i] = True

        center_batch = torch.from_numpy(centers).to(device=device)
        w_batch = torch.from_numpy(ws).to(device=device)
        h_batch = torch.from_numpy(hs).to(device=device)
        rad_batch = torch.from_numpy(radians).to(device=device)
        is_valid = torch.from_numpy(valids).to(device=device)

        return (center_batch, w_batch, h_batch, rad_batch), is_valid


    @staticmethod
    def checkEllipse_batch(xx, yy, centers, w, h, theta):
        x, y = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1), yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)
        cos_t, sin_t = torch.cos(theta).view(-1, 1, 1), torch.sin(theta).view(-1, 1, 1)
        x_rot, y_rot = x * cos_t + y * sin_t, -x * sin_t + y * cos_t
        return (x_rot / w.view(-1, 1, 1))**2 + (y_rot / h.view(-1, 1, 1))**2
    
    def EllipseConfidence_batch(self, pred, el_info):
        c, w, h, theta = el_info
        xx, yy = self._xx, self._yy
        x = xx.unsqueeze(0) - c[:, 0].view(-1, 1, 1)
        y = yy.unsqueeze(0) - c[:, 1].view(-1, 1, 1)

        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)
        xr = x * cos_t + y * sin_t
        yr = -x * sin_t + y * cos_t

        wv = w.view(-1, 1, 1).clamp_min(1e-6)
        hv = h.view(-1, 1, 1).clamp_min(1e-6)
        mask = (xr / wv) ** 2 + (yr / hv) ** 2 < 1.0

        masked = pred * mask
        conf = masked.sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8)
        return conf, mask
    
    
    @staticmethod
    def checkEllipse_batch(xx, yy, centers, w, h, theta):
        x, y = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1), yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)
        cos_t, sin_t = torch.cos(theta).view(-1, 1, 1), torch.sin(theta).view(-1, 1, 1)
        x_rot, y_rot = x * cos_t + y * sin_t, -x * sin_t + y * cos_t
        return (x_rot / w.view(-1, 1, 1))**2 + (y_rot / h.view(-1, 1, 1))**2


    def fit_ellipse_compact(self, tensor, threshold=0.5, mask=None):
        """Fitting an ellipse to the thresholded pixels which form the largest connected area.
        Args:
            tensor (3D torch tensor): batch x h x w, Prediction from the DeepVOG network (240, 320), float [0,1]
            threshold (scalar): thresholding pixels for fitting an ellipse
            mask (4D torch tensor): Prediction from DeepVOG-3D network for eyelid region (240, 320), float [0,1].
                                    intended for masking away the eyelid such as the fitting is better
        Returns:
            ellipse_info (tuple): A tuple of (center, w, h, radian), center is a list [x-coordinate, y-coordinate] of the ellipse centre. 
                                    None is returned if no ellipse can be found.
            confidence (1D torch tensor): Confidence of the fitted ellipse.
            n_pxls (1D torch tensor): Number of pixels used for fitting the ellipse.
            is_valid (1D torch tensor): Boolean tensor indicating if the ellipse is valid
        """
        # isolated_pred = isolate_islands(img, threshold = threshold)
        roi = tensor > threshold
        tensor = tensor * roi  # avoid in-place (safer in multithread)
        perim = self.bwperim_batch(roi)

        if mask is not None:
            perim = perim & mask

        el_info, is_valid = self.gen_ellipse_batch_info(perim, device=tensor.device)
        conf, el_masks = self.EllipseConfidence_batch(tensor, el_info)
        n_pxls = roi.sum(dim=(-2, -1)).float()
        is_valid &= (n_pxls > 0)

        return el_info, el_masks, conf, is_valid
    
    def process_region(self, pred, threshold, label, mask):
        ellipses, el_masks, conf, is_valid = self.fit_ellipse_compact(pred, threshold, mask)
        c, w, h, rad = ellipses
        return {
            f"{label}_center_x": c[:, 0],
            f"{label}_center_y": c[:, 1],
            f"{label}_w": w,
            f"{label}_h": h,
            f"{label}_radius": (w + h) * 0.5,
            f"{label}_radian": rad,
            f"{label}_confidence": conf,
        }, is_valid, el_masks
    
    def ellipse_fitting(self, frame_batch):
        start_time = time.time()
        self.batch_size = frame_batch['idxs'].shape[0]
        self.is_valid = frame_batch['is_valid']
        segs = frame_batch['segs']
        img_gray = frame_batch['imgs']
        sclera_masks = (segs[:, :, :, -1] > self.args['threshold_sclera'])

        el_pupil, is_valid_pupil, pupil_masks = self.process_region(segs[:, :, :, 0], self.args['threshold_pupil'], label='pupil', mask = None)
        el_iris, is_valid_iris, iris_masks = self.process_region(segs[:, :, :, 1], self.args['threshold_iris'], label='iris', mask = sclera_masks)
        el_dicts = {**el_pupil,**el_iris}
        self.is_valid &=  (is_valid_pupil != 0) & (is_valid_iris != 0)
        self.frame_counter += self.batch_size

        
                # seg_mask = (segs[:, :, :, 1] > 0.5) & (segs[:, :, :, 0] < 0.5) & \
        #         (segs[:, :, :, 3] > 0.5) & (segs[:, :, :, 2] < 0.5)
        seg_mask = (iris_masks & ~pupil_masks & sclera_masks)
        # seg_mask = (segs[:, :, :, 1] > self.params['threshold_iris']) & (segs[:, :, :, 0] < self.params['threshold_pupil']) & \
        #          (segs[:, :, :, -1] < self.params['threshold_sclera']) 
        useful_maps = torch.zeros_like(img_gray, dtype=torch.float32)
        useful_maps[seg_mask] = img_gray[seg_mask]
        blink_score = torch.sum(pupil_masks & sclera_masks, dim = [-2,-1])/ torch.sum(pupil_masks, dim = [-2,-1])
        blink_score[~self.is_valid] = torch.nan
        blink = blink_score < self.blink_threshold

        el_dicts['blink'] = blink
        el_dicts['is_valid'] = self.is_valid

        # Outputs
        gaze_batch = {
            'imgs': img_gray,
            'is_valid': self.is_valid,
            'ellipses': el_dicts,
            'idxs': frame_batch['idxs'],
            'blink': blink,
            'iris_masks': iris_masks,
            'pupil_masks': pupil_masks,
        }
        torsion_batch = {
            'useful_maps': useful_maps,
            'is_valid': self.is_valid,
            'ellipses': el_dicts,
            'idxs': frame_batch['idxs'],
            'blink': blink
        }
        self.elapsed_time += time.time() - start_time

        # put data on target queue(s)
        if self.use_queue:
            if self.args['gaze_tracking_flag']:
                self.threads['ques']['gaze_tracking'].put(gaze_batch)
            if self.args['torsion_tracking_flag'] and not(self.args['torsion_geometric_correction_type']=='3D'):
                self.threads['ques']['torsion_tracking'].put(torsion_batch)

            # el_dicts_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in el_dicts.items()}
            def to_np(v):
                if isinstance(v, torch.Tensor):
                    return v.detach().cpu().numpy()
                return np.asarray(v)

            # el_dicts is dict of batched tensors/arrays, shape (B,)
            el_np = {k: to_np(v) for k, v in el_dicts.items()}

            # infer batch size B
            first = next(iter(el_np.values()))
            B = int(first.shape[0]) if np.asarray(first).ndim > 0 else 1

            el_out = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                    for k, v in el_dicts.items()}
            self.threads['ques']['ellipse_out'].put(el_out)
                
            if self.args['seg_video_flag']: 
                bid = int(frame_batch['idxs'][0])  # or frame_batch.get('batch_id', frame_batch['idxs'][0])
                if self.args['write_seg_video_type'] == 'raw':
                    payload = segs.detach().cpu().numpy()
                elif self.args['write_seg_video_type'] == 'processed':
                    payload = seg_mask.detach().cpu().numpy()
                self.threads['ques']['segment_out'].put((bid, payload))
        else:
            return el_dicts, gaze_batch, torsion_batch
        
    
    def run(self):
        while True:
            # read data from source queue
            frame_batch = self.threads['ques']['ellipse_fitting'].get()
            if frame_batch is None: # poison pill
                self.threads['ques']['ellipse_out'].put(None)
                if self.args['extract_segment_map']:
                    self.threads['ques']['segment_out'].put(None)

                if self.args['gaze_tracking_flag']:
                    self.threads['ques']['gaze_tracking'].put(None)
                if self.args['torsion_tracking_flag'] and not(self.args['torsion_geometric_correction_type']=='3D'):
                    self.threads['ques']['torsion_tracking'].put(None)
                break
            else:
                self.ellipse_fitting(frame_batch)
