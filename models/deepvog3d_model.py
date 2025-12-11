import os
import torch
from .segmentation_model import Unet_3in4out_model, SegResNet_3in3out_model
import torch
import monai
from monai.transforms import (
    EnsureChannelFirst,   
    AsChannelLast,
    CastToType,
    Compose,
    Lambda,
    Resize,
    ScaleIntensity,
    ToTensor,   # Converts NumPy arrays to PyTorch tensors
)
from monai.transforms.compose import Transform

class Gray2Rgb(Transform):
    """
    Converts a gray image (a single color channel) to RGB (three color channels, identical to channel 0)
    and ensures the output is in the format (B, C, H, W).
    """
    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        if img.ndim == 3:  # If (B, H, W), add a channel dimension
            img = img.unsqueeze(-1)  # Shape: (B, H, W, 1)
        if img.ndim == 4 and img.shape[-1] == 1:  # If batch of single-channel images
            img = img.repeat(1, 1, 1, 3)  # Repeat the channel to get RGB
        elif img.ndim == 4 and img.shape[-1] == 3:  # If shape is already (B, H, W, 3)
            pass  # No action needed
        else:
            raise ValueError(f'Input img to Gray2Rgb needs to have 1 or 3 channels, got {img.shape[-1]}.')
        # Ensure the image has channels first: (B, C, H, W)
        return img


class Model_3DeepVOG:
    def __init__(self, device= 'cpu', 
                 model=None, 
                 ff_model_weights=None, 
                 video_width=320, 
                 video_height=240):
        
        if model is None:
            # model = Unet_3in4out_model()
            model = SegResNet_3in3out_model()
        if ff_model_weights is None:
            # fn_model_weights = 'berk_model_weights.pth'   
            # out_channel, in_channel = model.out_channels, model.in_channels
            fn_model_weights = 'SegResNet_weights.pth'
            out_channel, in_channel = 3, 3  # SegResNet outputs 3 channels (pupil, iris, visible eye)
            # load model weights
            base_dir = os.path.dirname(__file__)
            ff_model_weights = os.path.join(base_dir, fn_model_weights)
            state_dict = torch.load(ff_model_weights, map_location=device,  weights_only=True)
            model.load_state_dict(state_dict)
        else:
            original_state_dict = torch.load(ff_model_weights, map_location=device, weights_only=True)
            model.load_state_dict(original_state_dict)
            out_channel, in_channel = 3, 3
        model.eval()
        
        # set up transformation pipeline: forward and inverse
        sigmoid = torch.nn.Sigmoid()
        video_shape = (video_height, video_width)
        # out_channel, in_channel = model.out_channels, model.in_channels
        
        transforms = Compose(
            [
                Gray2Rgb(),
                ScaleIntensity(),
                Resize(spatial_size=(240, 320, 3)),  # Ensure the spatial size matches the image dimensions
                CastToType(torch.float32),
                ToTensor(),
            ]
            )
        transforms_inv_seg = Compose(
                [
                    Lambda(lambda im: sigmoid(im)),
                    AsChannelLast(channel_dim=1),
                    Resize(spatial_size=(video_height, video_width, out_channel)) # Resize to match video_shape
                ]
            )

        self.video_shape = video_shape
        self.ff_model_weights = ff_model_weights
        self.model = model
        self.device = device
        self.transforms = transforms
        self.transforms_inv_seg = transforms_inv_seg

    def get_model(self):
        '''
        Returns
        -------
        model: pytorch/monai 3D UNet/VNet model
            model can handled as any other torch.nn.Module

        '''
        return self.model
    
    def empty_gpu_cache(self):
        torch.cuda.empty_cache()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation maps for input images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, C) for RGB images or (B, H, W) for grayscale images.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, H, W, 4) with segmentation maps for:
            0. Pupil, 1. Iris, 2. Glints, 3. Visible map.
        """
        with torch.no_grad():
            x_trans = self.transforms(x).permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            y = self.transforms_inv_seg(self.model(x_trans))  # Shape: (B, H, W, out_model)
        return y
