import torch
import torch.nn as nn
import torch.nn.functional as F

from tactmorph.model import SpatialTransformer


class ConvUnit2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleUnit2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedFeatureEncoder2D(nn.Module):
    """
    Feature pyramid extractor used with shared weights for fixed and moving images.
    """

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.level0 = ConvUnit2D(in_channels, c1)
        self.down1 = DownsampleUnit2D(c1, c2)
        self.level1 = ConvUnit2D(c2, c2)
        self.down2 = DownsampleUnit2D(c2, c3)
        self.level2 = ConvUnit2D(c3, c3)
        self.down3 = DownsampleUnit2D(c3, c4)
        self.level3 = ConvUnit2D(c4, c4)

    def forward(self, image):
        f0 = self.level0(image)
        f1 = self.level1(self.down1(f0))
        f2 = self.level2(self.down2(f1))
        f3 = self.level3(self.down3(f2))
        return (f0, f1, f2, f3)


class FusionUnit2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvUnit2D(in_channels, out_channels),
            ConvUnit2D(out_channels, out_channels),
        )

    def forward(self, *features):
        return self.block(torch.cat(features, dim=1))


class RegistrationDecoder2D(nn.Module):
    """
    Coarse-to-fine decoder that predicts a flow at each feature resolution.
    """

    def __init__(self, base_channels=16):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.deep_fusion = FusionUnit2D(c4 * 2, c4)
        self.flow3 = nn.Conv2d(c4, 2, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec2 = FusionUnit2D(c3 + c3 * 2 + 2, c3)
        self.flow2_delta = nn.Conv2d(c3, 2, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec1 = FusionUnit2D(c2 + c2 * 2 + 2, c2)
        self.flow1_delta = nn.Conv2d(c2, 2, kernel_size=3, padding=1)

        self.up0 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec0 = FusionUnit2D(c1 + c1 * 2 + 2, c1)
        self.refine = nn.Sequential(
            ConvUnit2D(c1, c1),
            ConvUnit2D(c1, c1),
        )
        self.flow0_delta = nn.Conv2d(c1, 2, kernel_size=3, padding=1)

    @staticmethod
    def upsample_to(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def upsample_flow_to(flow, reference):
        src_h, src_w = flow.shape[-2:]
        dst_h, dst_w = reference.shape[-2:]
        if (src_h, src_w) == (dst_h, dst_w):
            return flow

        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)
        upsampled = F.interpolate(flow, size=(dst_h, dst_w), mode="bilinear", align_corners=True)
        scale = torch.tensor([scale_x, scale_y], dtype=upsampled.dtype, device=upsampled.device)
        return upsampled * scale.view(1, 2, 1, 1)

    def forward(self, moving_features, fixed_features):
        m0, m1, m2, m3 = moving_features
        f0, f1, f2, f3 = fixed_features

        x = self.deep_fusion(m3, f3)
        flow3 = self.flow3(x)

        flow3_up = self.upsample_flow_to(flow3, m2)
        x = self.dec2(self.upsample_to(self.up2(x), m2), m2, f2, flow3_up)
        flow2 = flow3_up + self.flow2_delta(x)

        flow2_up = self.upsample_flow_to(flow2, m1)
        x = self.dec1(self.upsample_to(self.up1(x), m1), m1, f1, flow2_up)
        flow1 = flow2_up + self.flow1_delta(x)

        flow1_up = self.upsample_flow_to(flow1, m0)
        x = self.dec0(self.upsample_to(self.up0(x), m0), m0, f0, flow1_up)
        x = self.refine(x)
        flow0 = flow1_up + self.flow0_delta(x)
        return [flow3, flow2, flow1, flow0]


class SharedFeatureTactMorph2D(nn.Module):
    """
    TactMorph variant with separated shared-weight feature extraction.

    The same encoder instance extracts features for moving and fixed images.
    The registration decoder receives the two feature pyramids and upsamples
    through levels to predict the dense sampling flow.
    """

    def __init__(self, base_channels=16):
        super().__init__()
        self.feature_encoder = SharedFeatureEncoder2D(in_channels=1, base_channels=base_channels)
        self.registration_decoder = RegistrationDecoder2D(base_channels=base_channels)
        self.transformer = SpatialTransformer()

    def forward(self, moving, fixed, return_pyramid=False):
        moving_features = self.feature_encoder(moving)
        fixed_features = self.feature_encoder(fixed)
        flow_pyramid = self.registration_decoder(moving_features, fixed_features)
        flow = flow_pyramid[-1]
        warped = self.transformer(moving, flow)
        if return_pyramid:
            return warped, flow, flow_pyramid
        return warped, flow
