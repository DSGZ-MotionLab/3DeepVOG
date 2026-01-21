import time
import threading
import numpy as np
import torch
import cv2
import kornia

class PostProcessing(threading.Thread):
    """
    Post-processing for segmentation masks.

    Args expected in `args`:
      - device: torch.device
      - threshold_pupil, threshold_iris, threshold_sclera: float
      - th_under_exposure, th_over_exposure: float
      - connected_components:
            False / None         -> no cleanup
            "morph"              -> GPU morphological cleanup (fast)
            "largest"            -> CPU largest connected component (exact, slower)
      - cc_channels: list of channel indices to clean (default: [0,1,-1])
      - morph_kernel: int (default 3)
      - morph_iters: int (default 1)
      - is_parallel: bool (queue mode)
    """

    def __init__(self, threads, args, daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.name = "Thread-PostProcessing"
        self.threads = threads
        self.params = args
        self.use_queue = args.get("is_parallel", use_queue)
        self.elapsed_time = 0.0

        self.device = self.params["device"]

        # Which channels to threshold/clean; default pupil, iris, sclera(last)
        self.int_channels = self.params.get("cc_channels", [0, 1, -1])

        # Prebuild thresholds on device (avoid per-batch tensor creation)
        self.seg_th = torch.tensor(
            [
                self.params["threshold_pupil"],
                self.params["threshold_iris"],
                self.params["threshold_sclera"],
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 1, -1)  # (1,1,1,Csel)

        # Morph config
        self.morph_kernel = int(self.params.get("morph_kernel", 3))
        self.morph_iters = int(self.params.get("morph_iters", 1))

    def run(self):
        while True:
            frame_batch = self.threads["ques"]["post_processing"].get()
            if frame_batch is None:  # poison pill
                self.threads["ques"]["ellipse_fitting"].put(None)
                break
            self.post_processing(frame_batch)

    def post_processing(self, frame_batch):
        t0 = time.time()

        segs = frame_batch["segs"]  # (B,H,W,C_all), float
        seg_sel = segs[..., self.int_channels]  # (B,H,W,Csel)

        # Threshold into boolean ROI
        seg_roi = seg_sel > self.seg_th  # bool (B,H,W,Csel)
        cc_mode = self.params.get("connected_components", False)

        if cc_mode == "largest":
            # exact largest CC per (batch, channel) on CPU
            seg_roi = self.apply_largest_cc_cv2(seg_roi)

        elif cc_mode == "morph":
            # fast GPU cleanup
            seg_roi = self.fast_morph(seg_roi, k=self.morph_kernel, iters=self.morph_iters)

        elif cc_mode:
            # backward-compat: if someone passes True, treat as "largest"
            seg_roi = self.apply_largest_cc_cv2(seg_roi)

        # Apply ROI back onto original logits/probs
        # seg_roi is bool -> multiply keeps original values inside ROI
        segs[..., self.int_channels] = seg_roi.to(seg_sel.dtype) * seg_sel
        frame_batch["segs"] = segs

        # Exposure validity
        eva_val = frame_batch["imgs"].mean(dim=(-2, -1))
        frame_batch["is_valid"] = (eva_val > self.params["th_under_exposure"]) & (
            eva_val < self.params["th_over_exposure"]
        )

        self.elapsed_time += time.time() - t0

        if self.use_queue:
            self.threads["ques"]["ellipse_fitting"].put(frame_batch)
        else:
            return frame_batch

    @staticmethod
    def fast_morph(seg_roi: torch.Tensor, k: int = 3, iters: int = 1) -> torch.Tensor:
        """
        seg_roi: (B,H,W,C) bool on GPU/CPU
        returns: (B,H,W,C) bool
        """
        device = seg_roi.device
        kernel = torch.ones((k, k), device=device, dtype=torch.float32)

        x = seg_roi.permute(0, 3, 1, 2).float()  # (B,C,H,W)

        # opening: erosion -> dilation (remove speckles)
        for _ in range(iters):
            x = kornia.morphology.erosion(x, kernel)
            x = kornia.morphology.dilation(x, kernel)

        # optional: closing (fill tiny holes). Uncomment if needed.
        # for _ in range(iters):
        #     x = kornia.morphology.dilation(x, kernel)
        #     x = kornia.morphology.erosion(x, kernel)

        return (x > 0.5).permute(0, 2, 3, 1)

    @staticmethod
    def apply_largest_cc_cv2(seg_roi: torch.Tensor) -> torch.Tensor:
        """
        seg_roi: (B,H,W,C) bool tensor (any device)
        returns: (B,H,W,C) bool tensor on original device
        """
        device = seg_roi.device
        seg_np = seg_roi.detach().cpu().numpy().astype(np.uint8)  # (B,H,W,C)
        B, H, W, C = seg_np.shape
        out = np.zeros_like(seg_np, dtype=np.uint8)

        for i in range(B):
            for ch in range(C):
                region = seg_np[i, :, :, ch]
                if region.sum() == 0:
                    continue
                n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=4)
                if n > 1:
                    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    out[i, :, :, ch] = (labels == k)

        return torch.from_numpy(out.astype(bool)).to(device=device)