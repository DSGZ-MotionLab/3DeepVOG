
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

import subprocess    #Run a command at python script  (e.g. ls, dir, etc.)
import threading
import queue
import plotly.offline as pyo
from concurrent.futures import ThreadPoolExecutor
# import skvideo.io as skv
from skimage.color import label2rgb
from skimage import measure
from astropy.convolution import convolve as nan_convolve
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
        # self.skimg_EllipseModel = measure.EllipseModel()

    def run(self):
        while True:
            # read data from source queue
            frame_batch = self.threads['ques']['ellipse_fitting'].get()
            if frame_batch is None: # poison pill
                self.threads['ques']['ellipse_out'].put(None)
                if self.args.get('extract_segment_map', False):
                    self.threads['ques']['segment_out'].put(None)
                # if self.args['viz_segmentation']:
                #     self.threads['ques']['ellipse_visualization'].put(None)
                if self.args['do_gaze_tracking']:
                    self.threads['ques']['gaze_tracking'].put(None)
                if self.args['do_torsion_tracking'] and not(self.args['torsion_geometric_correction_type']=='3D'):
                    self.threads['ques']['torsion_tracking'].put(None)
                break
            else:
                self.ellipse_fitting(frame_batch)
    
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
    def gen_ellipse_batch_info(perim, device, parallel=False):
        #TODO: -> can try to implement https://github.com/artuppp/EllipseFitCUDA
        perim_np = perim.cpu().numpy()   #-> slow down the process
        batch_size = perim_np.shape[0]
        # Helper function to process each ellipse
        def process_single_ellipse(i):
            vertices = np.column_stack(np.where(perim_np[i]))
            if vertices.shape[0] > 6:
                el_info = cv2.fitEllipse(vertices)
                center = [el_info[0][1], el_info[0][0]]
                w = el_info[1][0] / 2
                h = el_info[1][1] / 2
                radian = np.pi / 2 - np.deg2rad(el_info[2])
                return center, w, h, radian, True
            else:
                return [np.nan, np.nan], np.nan, np.nan, np.nan, False
        # Parallel or sequential ellipse fitting based on the argument
        if parallel:   #-> might be faster (depends on the number of ellipses)
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_single_ellipse, range(batch_size)))
        else:
            results = [process_single_ellipse(i) for i in range(batch_size)]
        # Unpack results
        centers, ws, hs, radians, valids = zip(*results)

        # Convert to tensors
        center_batch = torch.tensor(centers, dtype=torch.float32, device=device)
        w_batch = torch.tensor(ws, dtype=torch.float32, device=device)
        h_batch = torch.tensor(hs, dtype=torch.float32, device=device)
        radian_batch = torch.tensor(radians, dtype=torch.float32, device=device)
        is_valid = torch.tensor(valids, dtype=torch.bool, device=device)

        return (center_batch, w_batch, h_batch, radian_batch), is_valid


    @staticmethod
    def checkEllipse_batch(xx, yy, centers, w, h, theta):
        x, y = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1), yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)
        cos_t, sin_t = torch.cos(theta).view(-1, 1, 1), torch.sin(theta).view(-1, 1, 1)
        x_rot, y_rot = x * cos_t + y * sin_t, -x * sin_t + y * cos_t
        return (x_rot / w.view(-1, 1, 1))**2 + (y_rot / h.view(-1, 1, 1))**2
    
    @staticmethod
    def EllipseConfidence_batch(pred, el_info, device):
        c, w, h, theta = el_info
        yy, xx = torch.meshgrid(
            torch.arange(pred.shape[-2], device=device),
            torch.arange(pred.shape[-1], device=device),
            indexing='ij'
        )
        mask = EllipseFitting.checkEllipse_batch(xx, yy, c, w, h, theta) < 1
        masked = pred * mask
        return masked.sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8)
    
    
    @staticmethod
    def checkEllipse_batch(xx, yy, centers, w, h, theta):
        x, y = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1), yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)
        cos_t, sin_t = torch.cos(theta).view(-1, 1, 1), torch.sin(theta).view(-1, 1, 1)
        x_rot, y_rot = x * cos_t + y * sin_t, -x * sin_t + y * cos_t
        return (x_rot / w.view(-1, 1, 1))**2 + (y_rot / h.view(-1, 1, 1))**2
    
    @staticmethod
    def EllipseConfidence_batch(pred, el_info, device):
        c, w, h, theta = el_info
        yy, xx = torch.meshgrid(
            torch.arange(pred.shape[-2], device=device),
            torch.arange(pred.shape[-1], device=device),
            indexing='ij'
        )
        mask = EllipseFitting.checkEllipse_batch(xx, yy, c, w, h, theta) < 1
        masked = pred * mask
        return masked.sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8), mask.bool()


    @staticmethod
    def fit_ellipse_compact(tensor, threshold = 0.5, mask=None):
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
        device = tensor.device
        roi = tensor > threshold
        tensor[~roi] = 0.0   # set the pixels below threshold to 0
        perim_batch = EllipseFitting.bwperim_batch(roi)   #bust be binary!!
        # # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
        if mask is not None:
            perim_batch[~mask] = False
        ellipse_info, is_valid = EllipseFitting.gen_ellipse_batch_info(perim_batch, device=device, parallel=False)
        confidence, el_masks = EllipseFitting.EllipseConfidence_batch(tensor*roi, ellipse_info, device=device)
        n_pxls = torch.nansum(roi, dim=[-2, -1]).float()
        is_valid &= (n_pxls != 0)
        return ellipse_info, el_masks, confidence, is_valid
    
    @staticmethod
    def process_region(pred, threshold, label, mask):
        ellipses, el_masks, confidence, is_valid = EllipseFitting.fit_ellipse_compact(pred, threshold = threshold, mask=mask)
        return {
            f'{label}_center_x': ellipses[0][:, 0],
            f'{label}_center_y': ellipses[0][:, 1],
            f'{label}_w': ellipses[1],
            f'{label}_h': ellipses[2],
            f'{label}_radius': (ellipses[1] + ellipses[2]) / 2,
            f'{label}_radian': ellipses[3],
            f'{label}_confidence': confidence,
            # f'{label}_mask': ,
        }, is_valid, el_masks
    
    def ellipse_fitting(self, frame_batch):
        start_time = time.time()
        self.batch_size = frame_batch['idxs'].shape[0]
        self.is_valid = frame_batch['is_valid']
        segs = frame_batch['segs']
        img_gray = frame_batch['imgs']
        sclera_masks = (segs[:, :, :, -1] > self.args['threshold_sclera'])

        el_pupil, is_valid_pupil, pupil_masks = EllipseFitting.process_region(segs[:, :, :, 0], self.args['threshold_pupil'], label='pupil', mask = None)
        el_iris, is_valid_iris, iris_masks = EllipseFitting.process_region(segs[:, :, :, 1], self.args['threshold_iris'], label='iris', mask = sclera_masks)
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
            self.threads['ques']['ellipse_out'].put(el_dicts)
            # if self.args['viz_segmentation']:
            #     self.threads['ques']['ellipse_visualization'].put(frame_batch)
                
            if self.args.get('extract_segment_map', False):
                if self.args['extract_segment_map'] == "all": 
                    self.threads['ques']['segment_out'].put(segs) 
                elif self.args['extract_segment_map'] == "sclera":
                    self.threads['ques']['segment_out'].put(sclera_masks)
                elif self.args['extract_segment_map'] == "useful":
                    self.threads['ques']['segment_out'].put(useful_maps)
                # sclara_masks = (frame_batch['segs'][:,:,:,-1] > 0.5).bool()
                # pupil_masks = (frame_batch['segs'][:,:,:,0] > 0.5).bool()
                # iris_masks = (frame_batch['segs'][:,:,:,1] > 0.5).bool()
                # mask = torch.stack([pupil_masks, iris_masks, sclara_masks], dim=-1)
                # self.threads['ques']['segment_out'].put(mask)
                # self.threads['ques']['segment_out'].put(frame_batch)
            if self.args['do_gaze_tracking']:
                self.threads['ques']['gaze_tracking'].put(gaze_batch)
            if self.args['do_torsion_tracking'] and not(self.args['torsion_geometric_correction_type']=='3D'):
                self.threads['ques']['torsion_tracking'].put(torsion_batch)
        else:
            return el_dicts, gaze_batch, torsion_batch
