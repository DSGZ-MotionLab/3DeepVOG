import torch
import monai
# Global device
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

def Unet_3in4out_model():
    # input: 4 channels: rgb
    # output: 4 channels: 0: Pupil, 1: Iris, 2: Glints, 3: Sclera
    # 'best_metric_model_dv3d_segmentation2d_dict_withdropout.pth' follows this model's architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=4, 
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2), 
        dropout=0.5,
        num_res_units=2
    ).to(device)
    return model

def Unet_3in3out_model():
    """
    U-Net
    Advantages:
    - Simplicity: Easy to implement and train, works well for medical image segmentation.
    - Efficiency: Lightweight compared to attention-based or transformer-based models.
    - Strong Baseline: Performs well for many tasks with minimal tuning.
    Disadvantages:
    - Limited Context: Struggles with capturing long-range dependencies due to convolution-only architecture.
    - Scaling: Performance may degrade on complex datasets compared to newer architectures.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=0.5,
        num_res_units=2
    ).to(device)
    return model


def UnetPP_3in3out_model():
    """
    U-Net++ (Nested U-Net)
    Advantages:
    - Enhanced Feature Fusion: Incorporates nested skip connections for better multi-scale feature learning.
    - Improved Accuracy: Typically achieves better results than U-Net on small and complex datasets.
    Disadvantages:
    - Computational Cost: Higher memory and computation requirements due to additional paths.
    - Slower Inference: Nested paths increase processing time.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        kernel_size=3,
        up_kernel_size=3,
        act="RELU",
        norm="batch",
        dropout=0.5
    ).to(device)
    return model


def AttentionUnet_3in3out_model():
    """
    Attention U-Net
    Advantages:
    - Attention Mechanism: Improves focus on relevant features for segmentation.
    - Robust Performance: Handles class imbalance and noisy data better than U-Net.
    Disadvantages:
    - Computational Overhead: Attention layers increase computational complexity.
    - Sensitivity to Hyperparameters: Performance depends on careful tuning of attention weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.AttentionUnet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=0.5
    ).to(device)
    return model


def DynUNet_3in3out_model():
    """
    Dynamic U-Net (DynUNet)
    Advantages:
    - Flexibility: Dynamically adjusts the depth, kernel sizes, and strides based on input size.
    - Versatility: Handles varying image sizes and resolutions better than U-Net.
    - Stronger Baseline: Performs better on diverse datasets due to its adaptive nature.
    Disadvantages:
    - Complexity: More complex to configure compared to standard U-Net.
    - Resource Intensive: Requires careful tuning to avoid overfitting on small datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DynUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        kernel_size=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2],
        filters=(16, 32, 64, 128, 256),
        norm_name="instance"
    ).to(device)
    return model


def DeepLabV3Plus_3in3out_model():
    """
    DeepLabV3+
    Advantages:
    - Atrous Convolutions: Captures multi-scale context with dilated convolutions.
    - Strong Baseline: Excellent performance on complex datasets with large variability.
    - Pretrained Backbones: Leverages powerful pretrained CNNs like ResNet.
    Disadvantages:
    - Computational Overhead: Heavy model, especially with large backbones like ResNet101.
    - Requires Fine-Tuning: Sensitive to input size and dilation rates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DeepLabV3Plus(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        backbone="resnet50",
        pretrained=True,
        progress=True
    ).to(device)
    return model


def SegResNet_3in3out_model():
    """
    SegResNet
    Advantages:
    - Residual Connections: Helps mitigate vanishing gradient problems in deep networks.
    - Scalability: Handles both small and large datasets with minimal architecture changes.
    - Robust: Performs well in 2D and 3D segmentation tasks.
    Disadvantages:
    - Computational Overhead: Higher memory usage due to residual blocks.
    - Requires Fine-Tuning: Performance sensitive to normalization and learning rates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.SegResNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        init_filters=16,
        norm="batch",
        dropout_prob=0.2
    ).to(device)
    return model


def SegResNetVAE_3in3out_model():
    from monai.networks.nets import SegResNetVAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegResNetVAE(
        input_image_size=(240, 320),
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        init_filters=16,
        norm="batch",
        dropout_prob=0.2,
        vae_estimate_std=True  # or False depending on your use case
    ).to(device)
    return model


class SegFormerB0_3in3out(nn.Module):
    def __init__(self, input_size=(240, 320), out_channels=3, weight_path=None, device="cuda"):
        super().__init__()
        self.input_size = input_size

        # 1) Create base model WITHOUT changing num_labels yet
        base = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        # 2) Replace classifier head to 3 classes
        base.decode_head.classifier = nn.Conv2d(
            in_channels=base.config.decoder_hidden_size,
            out_channels=out_channels,
            kernel_size=1,
        )
        base.config.num_labels = out_channels
        base.config.image_size = input_size

        self.model = base.to(device).eval()

        # 3) Load your trained weights AFTER replacing head
        if weight_path is not None:
            sd = torch.load(weight_path, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            print("Loaded weights. Missing:", missing)
            print("Unexpected:", unexpected)

    def forward(self, x):
        out = self.model(pixel_values=x).logits
        return torch.nn.functional.interpolate(out, size=self.input_size, mode="bilinear", align_corners=False)
    
    @classmethod
    def read_model(cls, input_size=(240, 320), in_channels=3, out_channels=3, device="cuda"):
        model = cls(input_size=input_size, in_channels=in_channels, out_channels=out_channels)
        return model.to(device)
    
# ====== Read model, Loss Function, Optimizer, Loss function and Metrics =======
# model = MultiTaskNet_SegFormerB0.read_model().to(device)
# model = SegFormerB0_3in3out.read_model().to('cuda')
# weight_path = r"D:\jzhao\DeepVOG-project\result\segmentation\trained_model\2025-07-04_SegResNet_3in3out\best_model_2025-07-05_0.9086.pth"
# weight_path = r"D:\jzhao\DeepVOG-project\result\segmentation\trained_model\2025-07-08_SegFormerB0_3in3out\best_model_2025-07-12_0.9482.pth"
# # ff_model_weights = torch.load(weight_path, map_location='cuda', weights_only=True)
# state_dict = torch.load(weight_path, map_location='cuda',  weights_only=True)
# model.load_state_dict(state_dict)
# stripped_state_dict = strip_model_prefix(original_state_dict)
# model.load_state_dict(stripped_state_dict)