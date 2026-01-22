import os
import torch
import torch.nn.functional as F
from ..models.segmentation_model import Unet_3in4out_model, SegResNet_3in3out_model, SegFormerB0_3in3out


class Model_3DeepVOG:
    def __init__(
        self,
        device="cpu",
        model=None,
        ff_model_weights=None,
        video_width=320,
        video_height=240,
        in_size=(240, 320),  # network input size (H, W)
    ):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self.in_h, self.in_w = map(int, in_size)

        # ---- build model instance ----
        if model is None:
            net = SegResNet_3in3out_model()
        else:
            if model == "Unet_3in4out":
                net = Unet_3in4out_model()
            elif model == "SegResNet_3in3out":
                net = SegResNet_3in3out_model()
            elif model == "SegFormerB0_3in3out":
                net = SegFormerB0_3in3out()
            else:
                raise ValueError(f"Unsupported model type: {model}")

        # ---- load weights (CPU -> move to device) ----
        if ff_model_weights is None:
            base_dir = os.path.dirname(__file__)
            ff_model_weights = os.path.join(base_dir, "SegResNet_weights.pth")

        state_dict = torch.load(ff_model_weights, map_location="cpu")
        net.load_state_dict(state_dict, strict=True)
        net.eval().to(self.device)

        self.model = net
        self.ff_model_weights = ff_model_weights

        # reuse sigmoid module (tiny, but avoids re-creating)
        self._sigmoid = torch.nn.Sigmoid()

    def empty_gpu_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _to_bchw_rgb01(x: torch.Tensor) -> torch.Tensor:
        """
        Input:
          - (B,H,W) or (B,H,W,1) or (B,H,W,3) or (B,1,H,W) or (B,3,H,W)
        Output:
          - (B,3,H,W) float32 in [0,1] (approx)
        """
        if not torch.is_tensor(x):
            raise TypeError("predict() expects a torch.Tensor")

        # ensure float32
        if x.dtype != torch.float32:
            x = x.float()

        # shape to BCHW
        if x.ndim == 3:  # (B,H,W)
            x = x.unsqueeze(1)  # (B,1,H,W)
        elif x.ndim == 4:
            # could be BHWC or BCHW
            if x.shape[1] in (1, 3):  # BCHW
                pass
            elif x.shape[-1] in (1, 3):  # BHWC -> BCHW
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"Unsupported 4D shape for x: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported ndim for x: {x.ndim}")

        # grayscale -> RGB
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # ScaleIntensity() equivalent:
        # If input already in [0,1] nothing changes; if [0,255], normalize.
        # We do a cheap heuristic: if max>1.5, assume 0..255.
        if x.max().item() > 1.5:
            x = x / 255.0

        return x

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, H, W, 3) float tensor on CPU (same as your old code intent).
        """
        # prepare input
        x = self._to_bchw_rgb01(x)

        # move to model device/dtype
        p = next(self.model.parameters())
        x = x.to(device=p.device, dtype=p.dtype, non_blocking=True)

        # resize to net input
        x_in = F.interpolate(x, size=(self.in_h, self.in_w), mode="bilinear", align_corners=False)

        # forward
        y = self.model(x_in)  # (B, C, in_h, in_w)

        # sigmoid + resize back to video size
        y = self._sigmoid(y)
        y = F.interpolate(y, size=(self.video_height, self.video_width), mode="bilinear", align_corners=False)

        # output BHWC like you had
        y = y.permute(0, 2, 3, 1).contiguous()

        return y.cpu()