"""
EllipseFitting thread

This thread takes segmentation probability maps (pupil / iris / sclera) and fits ellipses
for pupil and iris per frame (batched). It also produces masks used downstream for:

- Eyeball model fitting / gaze tracking (GazeTracker)
- Iris torsion tracking (TorsionTracker)
- Optional overlay/segmentation video writing
- Saving ellipse parameters to disk (ellipse_out)

Design notes
------------
- The geometric ellipse fit itself uses OpenCV (cv2.fitEllipse), so it runs on CPU.
  Everything else (thresholding, perimeter extraction, confidence computation) is done
  with PyTorch and can run on GPU, but the CPU hop for fitEllipse is the bottleneck.

- Input assumptions:
  frame_batch contains:
    imgs: (B,H,W) float32 in [0,1]
    segs: (B,H,W,C) float32 in [0,1] (at least channels: pupil, iris, ..., sclera as last channel)
    is_valid: (B,) bool
    idxs: (B,) frame indices

Outputs
-------
1) gaze_batch -> to queue 'gaze_tracking' (if gaze_tracking_flag)
   Contains gray frames + ellipse params + pupil/iris masks + blink flags.

2) torsion_batch -> to queue 'torsion_tracking' (if torsion_tracking_flag and correction != '3D')
   Contains "useful_maps" (iris ring texture), ellipses, blink flags, etc.

3) ellipse_out -> to queue 'ellipse_out'
   A list[dict] with per-frame scalar/array ellipse outputs, for disk writer.

4) segment_out -> to queue 'segment_out' (optional)
   Raw seg volume or processed seg_mask for overlay video.

Key computations
----------------
- bwperim_batch: computes a thin perimeter mask around thresholded blobs (GPU-friendly).
- cv2.fitEllipse: ellipse parameter estimation from perimeter pixels (CPU).
- EllipseConfidence_batch: score how well the segmentation mass lies inside fitted ellipse.
- blink_score: overlap of pupil mask with sclera mask used to detect blinks.
"""

import os
import torch
import time
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class EllipseFitting(threading.Thread):
    def __init__(self, threads, args, daemon=False, use_queue=True, device="cpu"):
        super().__init__(daemon=daemon)
        self.name = "Thread-EllipseFitting"
        self.args = args
        self.threads = threads
        # If running the full pipeline in parallel, modules communicate via queues.
        self.use_queue = args.get("is_parallel", use_queue)
        self.device = args.get("device", device)

        # Blink threshold uses pupil mask overlap heuristic.
        self.blink_threshold = args.get("blink_threshold", 0.735)

        self.frame_counter = 0
        self.elapsed_time = 0

    @staticmethod
    def bwperim_batch(bw, n=4, mask=None):
        """
        Perimeter extraction (batch).

        A pixel is perimeter if it is 1 and at least one neighbor (4- or 8-connected)
        is 0. Uses padding+shifts (GPU-friendly).

        Args:
            bw: (B,H,W) bool tensor
            n: 4 or 8 connectivity
            mask: optional (B,H,W) bool to suppress perimeter pixels (e.g., eyelid mask)

        Returns:
            perim: (B,H,W) bool tensor
        """
        if n not in (4, 8):
            raise ValueError("bwperim_batch: n must be 4 or 8")

        padded = torch.nn.functional.pad(bw, (1, 1, 1, 1), mode="constant", value=0)

        north = padded[:, :-2, 1:-1]
        south = padded[:, 2:, 1:-1]
        west  = padded[:, 1:-1, :-2]
        east  = padded[:, 1:-1, 2:]

        idx = (north == bw) & (south == bw) & (west == bw) & (east == bw)

        if n == 8:
            ne = padded[:, :-2, 2:]
            nw = padded[:, :-2, :-2]
            se = padded[:, 2:, 2:]
            sw = padded[:, 2:, :-2]
            idx &= (ne == bw) & (nw == bw) & (se == bw) & (sw == bw)

        perim = (~idx) & bw

        # Remove image boundary to avoid artifacts
        perim[:, 0, :] = False
        perim[:, -1, :] = False
        perim[:, :, 0] = False
        perim[:, :, -1] = False

        if mask is not None:
            perim = perim & mask

        return perim

    @staticmethod
    def gen_ellipse_batch_info(perim, device, parallel=False):
        """
        Fit ellipses per image in the batch using OpenCV.

        NOTE: This hops to CPU numpy. This is the slow step.

        Args:
            perim: (B,H,W) bool tensor
            device: torch device for output tensors
            parallel: use ThreadPoolExecutor for per-frame CPU ellipse fits

        Returns:
            ellipse_info: (center_batch, w_batch, h_batch, radian_batch)
            is_valid: (B,) bool tensor
        """
        perim_np = perim.detach().cpu().numpy()  # (B,H,W)
        B = perim_np.shape[0]

        def fit_one(i):
            verts = np.column_stack(np.where(perim_np[i]))
            if verts.shape[0] > 6:
                el_info = cv2.fitEllipse(verts)  # (center(x,y), (MA,ma), angle_deg)
                # Your convention: store center as [x,y] in image coordinates
                center = [el_info[0][1], el_info[0][0]]  # swap because verts uses (row,col)
                w = el_info[1][0] / 2.0
                h = el_info[1][1] / 2.0
                rad = np.pi / 2 - np.deg2rad(el_info[2])
                return center, w, h, rad, True
            return [np.nan, np.nan], np.nan, np.nan, np.nan, False

        if parallel:
            with ThreadPoolExecutor() as ex:
                results = list(ex.map(fit_one, range(B)))
        else:
            results = [fit_one(i) for i in range(B)]

        centers, ws, hs, radians, valids = zip(*results)

        center_batch  = torch.tensor(centers, dtype=torch.float32, device=device)
        w_batch       = torch.tensor(ws, dtype=torch.float32, device=device)
        h_batch       = torch.tensor(hs, dtype=torch.float32, device=device)
        radian_batch  = torch.tensor(radians, dtype=torch.float32, device=device)
        is_valid      = torch.tensor(valids, dtype=torch.bool, device=device)

        return (center_batch, w_batch, h_batch, radian_batch), is_valid

    @staticmethod
    def checkEllipse_batch(xx, yy, centers, w, h, theta):
        """
        Compute ellipse equation value for each pixel, batched.

        Returns:
            val: (B,H,W) where val < 1 is inside ellipse.
        """
        x = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1)
        y = yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)

        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)

        xr = x * cos_t + y * sin_t
        yr = -x * sin_t + y * cos_t

        return (xr / w.view(-1, 1, 1)) ** 2 + (yr / h.view(-1, 1, 1)) ** 2

    @staticmethod
    def EllipseConfidence_batch(pred, el_info, device):
        """
        Confidence = fraction of prediction mass that lies inside the fitted ellipse.

        Args:
            pred: (B,H,W) float (segmentation probability)
            el_info: (center, w, h, theta)

        Returns:
            conf: (B,) float
            mask: (B,H,W) bool ellipse interior mask
        """
        c, w, h, theta = el_info

        yy, xx = torch.meshgrid(
            torch.arange(pred.shape[-2], device=device),
            torch.arange(pred.shape[-1], device=device),
            indexing="ij",
        )

        mask = EllipseFitting.checkEllipse_batch(xx, yy, c, w, h, theta) < 1
        masked = pred * mask
        conf = masked.sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8)
        return conf, mask.bool()

    @staticmethod
    def fit_ellipse_compact(tensor, threshold=0.5, mask=None):
        """
        Fit ellipse for each batch element.

        Steps:
        - Threshold prediction -> roi
        - Perimeter extraction on roi
        - Optional masking (e.g. remove eyelid)
        - cv2.fitEllipse on perimeter pixels
        - Confidence score inside ellipse

        Returns:
            ellipse_info: (center, w, h, radian)
            el_masks: (B,H,W) bool ellipse interior mask
            confidence: (B,) float
            is_valid: (B,) bool
        """
        device = tensor.device
        roi = tensor > threshold

        # Avoid modifying original tensor outside this function
        pred = tensor * roi

        perim = EllipseFitting.bwperim_batch(roi, mask=mask)
        ellipse_info, is_valid = EllipseFitting.gen_ellipse_batch_info(perim, device=device, parallel=False)

        confidence, el_masks = EllipseFitting.EllipseConfidence_batch(pred, ellipse_info, device=device)

        n_pxls = roi.sum(dim=(-2, -1)).float()
        is_valid &= (n_pxls != 0)
        return ellipse_info, el_masks, confidence, is_valid

    @staticmethod
    def process_region(pred, threshold, label, mask):
        """
        Helper for pupil/iris: fit ellipse and format outputs into a dict of batched tensors.
        """
        ellipses, el_masks, confidence, is_valid = EllipseFitting.fit_ellipse_compact(pred, threshold=threshold, mask=mask)
        center, w, h, rad = ellipses
        return {
            f"{label}_center_x": center[:, 0],
            f"{label}_center_y": center[:, 1],
            f"{label}_w": w,
            f"{label}_h": h,
            f"{label}_radius": (w + h) / 2,
            f"{label}_radian": rad,
            f"{label}_confidence": confidence,
        }, is_valid, el_masks

    def ellipse_fitting(self, frame_batch):
        """
        Main step:
        - Fit pupil+iris ellipses.
        - Compute blink flag and "useful_maps" for torsion.
        - Dispatch to downstream queues + save ellipse_out/segment_out.
        """
        start_time = time.time()
        self.batch_size = frame_batch["idxs"].shape[0]
        self.is_valid = frame_batch["is_valid"]

        segs = frame_batch["segs"]      # (B,H,W,C)
        img_gray = frame_batch["imgs"]  # (B,H,W)

        sclera_masks = (segs[:, :, :, -1] > self.args["threshold_sclera"])

        el_pupil, is_valid_pupil, pupil_masks = EllipseFitting.process_region(
            segs[:, :, :, 0], self.args["threshold_pupil"], label="pupil", mask=None
        )
        el_iris, is_valid_iris, iris_masks = EllipseFitting.process_region(
            segs[:, :, :, 1], self.args["threshold_iris"], label="iris", mask=sclera_masks
        )

        el_dicts = {**el_pupil, **el_iris}

        # Final validity combines exposure validity + successful ellipse fits
        self.is_valid &= (is_valid_pupil != 0) & (is_valid_iris != 0)
        self.frame_counter += self.batch_size

        # Iris ring region used for torsion: inside iris ellipse, outside pupil ellipse, and within sclera mask
        seg_mask = (iris_masks & ~pupil_masks & sclera_masks)

        useful_maps = torch.zeros_like(img_gray, dtype=torch.float32)
        useful_maps[seg_mask] = img_gray[seg_mask]

        # Blink heuristic: how much pupil mask overlaps sclera mask
        blink_score = torch.sum(pupil_masks & sclera_masks, dim=[-2, -1]) / (torch.sum(pupil_masks, dim=[-2, -1]) + 1e-8)
        blink_score[~self.is_valid] = torch.nan
        blink = blink_score < self.blink_threshold

        el_dicts["blink"] = blink
        el_dicts["is_valid"] = self.is_valid

        gaze_batch = {
            "imgs": img_gray,
            "is_valid": self.is_valid,
            "ellipses": el_dicts,
            "idxs": frame_batch["idxs"],
            "blink": blink,
            "iris_masks": iris_masks,
            "pupil_masks": pupil_masks,
        }
        torsion_batch = {
            "useful_maps": useful_maps,
            "is_valid": self.is_valid,
            "ellipses": el_dicts,
            "idxs": frame_batch["idxs"],
            "blink": blink,
        }

        self.elapsed_time += time.time() - start_time

        if not self.use_queue:
            return el_dicts, gaze_batch, torsion_batch

        # -----------------------
        # Send to downstream stages
        # -----------------------
        if self.args["gaze_tracking_flag"]:
            self.threads["ques"]["gaze_tracking"].put(gaze_batch)

        # For torsion correction type "3D", torsion uses gaze info and is triggered later.
        if self.args["torsion_tracking_flag"] and (self.args["torsion_geometric_correction_type"] != "3D"):
            self.threads["ques"]["torsion_tracking"].put(torsion_batch)

        # -----------------------
        # Save ellipses output (list[dict], one per frame)
        # -----------------------
        def to_np(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy()
            return np.asarray(v)

        el_np = {k: to_np(v) for k, v in el_dicts.items()}
        B = int(next(iter(el_np.values())).shape[0])

        el_list = []
        for i in range(B):
            row = {}
            for k, arr in el_np.items():
                a = np.asarray(arr)
                if a.ndim == 0:
                    row[k] = float(a)
                else:
                    vi = np.asarray(a[i])
                    row[k] = float(vi.reshape(-1)[0]) if vi.size == 1 else vi
            el_list.append(row)

        self.threads["ques"]["ellipse_out"].put(el_list)

        # -----------------------
        # Optional seg video payload
        # -----------------------
        if self.args["seg_video_flag"]:
            bid = int(frame_batch["idxs"][0])
            if self.args["write_seg_video_type"] == "raw":
                payload = segs.detach().cpu().numpy()
            elif self.args["write_seg_video_type"] == "processed":
                payload = seg_mask.detach().cpu().numpy()
            else:
                payload = segs.detach().cpu().numpy()
            self.threads["ques"]["segment_out"].put((bid, payload))

    def run(self):
        """
        Thread loop:
        - Consume frame_batch from 'ellipse_fitting'
        - Run ellipse_fitting()
        - On None: propagate poison pills to downstream queues
        """
        while True:
            frame_batch = self.threads["ques"]["ellipse_fitting"].get()
            if frame_batch is None:
                self.threads["ques"]["ellipse_out"].put(None)

                if self.args['seg_video_flag']:
                    self.threads['ques']['segment_out'].put(None)

                if self.args['gaze_tracking_flag']:
                    self.threads['ques']['gaze_tracking'].put(None)
                if self.args['torsion_tracking_flag'] and not(self.args['torsion_geometric_correction_type']=='3D'):
                    self.threads['ques']['torsion_tracking'].put(None)
                break
            else:
                self.ellipse_fitting(frame_batch)
