
#%% script_05_dv3d_threaded_classes.py
import os, torch, time, cv2, threading
# import sys    # sys.path.append("D:/git/DeepVOG3DTorch/DeepVOG/deepvog3D")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from monai.transforms import Resize
import skvideo.io as skv
import kornia.enhance as kornia_enhance
import plotly.offline as pyo
# from itertools import product
# import skvideo.io as skv
from skimage.color import label2rgb
from skimage import measure
from astropy.convolution import convolve as nan_convolve
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import label
# from scipy.spatial import ConvexHull

class PostProcessing(threading.Thread):
    def __init__(self, threads, args, daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.name = 'Thread-PostProcessing'
        self.threads = threads
        self.params = args
        self.use_queue = args.get('is_parallel', use_queue)  #priority to args['is_parallel'] if it exists
        self.elapsed_time = 0

    def run(self):
        while True:
            frame_batch = self.threads['ques']['post_processing'].get()
            if frame_batch is None: # poison pill
                self.threads['ques']['ellipse_fitting'].put(None)    
                break
            else:
                self.post_processing(frame_batch)
    
    def post_processing(self, frame_batch):
        time00 = time.time()
        batch_size = frame_batch['idxs'].shape[0]
        device = self.params['device']
        int_channels = [0, 1, -1]
        seg_th = torch.tensor([
            self.params['threshold_pupil'],
            self.params['threshold_iris'],
            self.params['threshold_sclera']
        ], device=device)
        seg_th_expand = seg_th.view(1, 1, 1, -1)
        segs = frame_batch['segs']
        seg_ROI = (segs[..., int_channels] > seg_th_expand)

        if self.params['connected_components']:
            seg_ROI = PostProcessing.apply_connected_components(seg_ROI, parallel=False)
            # segs[:, int_channels,:,:] *= seg_ROI
            # binary_pred = (segs[..., int_channels] > seg_th_expand).int()
            # Re-apply to original segs
            # frame_batch['segs'][..., int_channels] = (seg_ROI[..., int_channels] > seg_th_expand).int()
            frame_batch['segs'][..., int_channels] = seg_ROI * segs[..., int_channels]
        eva_val = frame_batch['imgs'].mean(dim=[-2, -1])
        frame_batch['is_valid'] = (eva_val > self.params['th_under_exposure']) & (eva_val < self.params['th_over_exposure'])
        self.elapsed_time += (time.time() - time00)
        if self.use_queue:
            self.threads['ques']['ellipse_fitting'].put(frame_batch)
        else:
            return frame_batch

    @staticmethod
    def apply_connected_components(seg_ROI_tensor, parallel=True):
        device = seg_ROI_tensor.device
        seg_np = seg_ROI_tensor.cpu().numpy()
        batch, H, W, C = seg_np.shape
        output = np.zeros_like(seg_np, dtype=bool)
        # Pre-filter empty masks to avoid unnecessary processing
        non_empty = np.any(seg_np, axis=(1, 2))
        tasks = [(i, ch) for i in range(batch) for ch in range(C) if non_empty[i, ch]]
        def process_mask(i, ch):
            region = seg_np[i, :, :, ch].astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=4)
            if n > 1:
                output[i, :, :, ch] = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if parallel:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1, prefer='threads', batch_size=10)(
                delayed(process_mask)(i, ch) for i, ch in tasks
            )
        else:
            for i, ch in tasks:
                process_mask(i, ch)
        return torch.from_numpy(output).to(device)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        #TODO: 
        # === implement pytorch compatable connected component algorithm + batch-processing ===
        # - kornia_contrib.connected_components  (speed is slower? the strange output)
        # - cc_torch module: https://github.com/zsef123/Connected_components_PyTorch/blob/main/example.ipynb
        # - graph connected component:https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/largest_connected_components.html
        # - connected-components-3d (for 3D image or batch-process?): https://pypi.org/project/connected-components-3d/1.0.1/
        
        #=== increase the computational speed ===
        # - cupy (only cuda gpu)
        # - cython (tried, the speed didn't improve for some reason...)
        
            # if self.device.type == 'cuda':
            #     for ch in range(len(int_channels)):
            #         regions = kornia_contrib.connected_components(seg_ROI[:,ch,...].unsqueeze(1).float()).to(torch.int64)
            #         # regions = measure.label(seg_ROI[:,ch,...].astype(np.int32), connectivity=None).astype(np.int32)  
            #         if regions.max() != 0:
            #             largest_cc = regions == (np.argmax(torch.bincount(regions.flatten())[1:]) + 1)
            #             # seg_ROI_np_cpu[idx,:,:,ch][~largest_cc] = False # set all other regions to zero