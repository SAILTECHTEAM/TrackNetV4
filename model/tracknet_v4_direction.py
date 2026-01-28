import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalMotionPrompt(nn.Module):
    """
    Extract direction-aware motion attention from consecutive frame differences.
    Separates positive (brightening) and negative (darkening) motion changes.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))  # learnable slope
        self.b = nn.Parameter(torch.randn(1))  # learnable bias

    def forward(self, x):
        """
        Args: x [B,3,3,H,W] - batch of 3 consecutive frames
        Returns:
            pos_maps [B,2,H,W]: Attention maps for positive changes
            neg_maps [B,2,H,W]: Attention maps for negative changes
        """
        B, T, C, H, W = x.shape
        gray = x.mean(dim=2)  # convert to grayscale [B,3,H,W]

        pos_maps = []
        neg_maps = []

        # compute frame differences: [t+1] - [t]
        for i in range(T - 1):
            diff = gray[:, i + 1] - gray[:, i]

            # Split into positive and negative components
            pos = F.relu(diff)  # Positive changes (brightening)
            neg = F.relu(-diff)  # Negative changes (darkening)

            # Convert to attention weights using sigmoid
            # Note: We use the same a,b parameters for stability, but could be separate
            pos_att = torch.sigmoid(self.a * pos + self.b)
            neg_att = torch.sigmoid(self.a * neg + self.b)

            pos_maps.append(pos_att)
            neg_maps.append(neg_att)

        # Stack temporal dimension
        pos_stack = torch.stack(pos_maps, dim=1)  # [B,2,H,W]
        neg_stack = torch.stack(neg_maps, dim=1)  # [B,2,H,W]

        return pos_stack, neg_stack


class DirectionalMotionFusion(nn.Module):
    """
    Fuse visual features with direction-aware motion attention.
    Uses learnable weights to combine positive and negative motion.
    """

    def __init__(self):
        super().__init__()
        # Initialize weights to 0.5 to start with balanced attention
        self.pos_weight = nn.Parameter(torch.tensor(0.5))
        self.neg_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, visual, pos_motion, neg_motion):
        """
        Args:
            visual: [B,3,H,W] - RGB visual features
            pos_motion: [B,2,H,W] - Positive motion attention
            neg_motion: [B,2,H,W] - Negative motion attention
        Returns: [B,3,H,W] - motion-enhanced features
        """
        # Combine positive and negative motion with learnable weights
        # Since pos and neg are pixel-wise mutually exclusive (mostly),
        # this acts as a selector

        # Motion 0: applied to Green channel (visual[:, 1])
        motion_0 = (
            self.pos_weight * pos_motion[:, 0] + self.neg_weight * neg_motion[:, 0]
        )

        # Motion 1: applied to Blue channel (visual[:, 2])
        motion_1 = (
            self.pos_weight * pos_motion[:, 1] + self.neg_weight * neg_motion[:, 1]
        )

        return torch.stack(
            [
                visual[:, 0],  # R channel unchanged
                motion_0 * visual[:, 1],  # G enhanced by combined motion_0
                motion_1 * visual[:, 2],  # B enhanced by combined motion_1
            ],
            dim=1,
        )


class TrackNetV4Direction(nn.Module):
    """
    Direction-Aware Motion-Enhanced U-Net (TrackNet V4 Variant)

    Improvements over standard TrackNet V4:
    - Separates motion into positive (brightening) and negative (darkening) changes
    - Learnable weights for positive vs negative motion importance
    - Helps distinguish object direction and contrast changes
    """

    def __init__(self):
        super().__init__()

        # motion processing modules
        self.motion_prompt = DirectionalMotionPrompt()
        self.motion_fusion = DirectionalMotionFusion()

        # encoder path: progressive downsampling
        self.enc1 = self._conv_block(9, 64, 2)  # 9->64
        self.enc2 = self._conv_block(64, 128, 2)  # 64->128
        self.enc3 = self._conv_block(128, 256, 3)  # 128->256
        self.enc4 = self._conv_block(256, 512, 3)  # 256->512 (bottleneck)

        # decoder path: progressive upsampling with skip connections
        self.dec1 = self._conv_block(768, 256, 3)  # 512+256->256
        self.dec2 = self._conv_block(384, 128, 2)  # 256+128->128
        self.dec3 = self._conv_block(192, 64, 2)  # 128+64->64

        # utility layers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.output = nn.Conv2d(64, 3, 1)

    def _conv_block(self, in_ch, out_ch, n_layers):
        """Create convolution block with batch norm and ReLU"""
        layers = []
        for i in range(n_layers):
            ch_in = in_ch if i == 0 else out_ch
            layers.extend(
                [
                    nn.Conv2d(ch_in, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args: x [B,9,288,512] - concatenated 3 frames (3*3=9 channels)
        Returns: [B,3,288,512] - probability heatmap for object detection
        """
        B = x.size(0)

        # extract directional motion attention
        # x.view splits the 9 channels back into 3 frames of 3 channels
        pos_motion, neg_motion = self.motion_prompt(x.view(B, 3, 3, 288, 512))

        # encoder path with skip connections
        e1 = self.enc1(x)  # [B,64,288,512]
        e1_pool = self.pool(e1)  # [B,64,144,256]

        e2 = self.enc2(e1_pool)  # [B,128,144,256]
        e2_pool = self.pool(e2)  # [B,128,72,128]

        e3 = self.enc3(e2_pool)  # [B,256,72,128]
        e3_pool = self.pool(e3)  # [B,256,36,64]

        bottleneck = self.enc4(e3_pool)  # [B,512,36,64]

        # decoder path with skip connections
        d1 = self.upsample(bottleneck)  # [B,512,72,128]
        d1 = self.dec1(torch.cat([d1, e3], dim=1))  # [B,256,72,128]

        d2 = self.upsample(d1)  # [B,256,144,256]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B,128,144,256]

        d3 = self.upsample(d2)  # [B,128,288,512]
        d3 = self.dec3(torch.cat([d3, e1], dim=1))  # [B,64,288,512]

        # generate base output and enhance with directional motion
        visual_output = self.output(d3)  # [B,3,288,512]
        enhanced_output = self.motion_fusion(visual_output, pos_motion, neg_motion)

        return torch.sigmoid(enhanced_output)


if __name__ == "__main__":
    # model initialization and testing
    model = TrackNetV4Direction()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TrackNetV4Direction initialized with {total_params:,} parameters")

    # forward pass test
    test_input = torch.randn(2, 9, 288, 512)
    test_output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("✓ TrackNetV4Direction ready for training!")
