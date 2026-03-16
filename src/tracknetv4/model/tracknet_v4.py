import torch
import torch.nn as nn


class MotionPrompt(nn.Module):
    """Extract motion attention from consecutive frame differences"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))  # learnable slope
        self.b = nn.Parameter(torch.randn(1))  # learnable bias

    def forward(self, x):
        """
        Args: x [B,T,3,H,W] - batch of T consecutive frames
        Returns: motion_features [B,T-1,3,H,W], motion_mask [B,T-1,H,W]
        """
        B, T, C, H, W = x.shape
        gray = x.mean(dim=2)  # convert to grayscale [B,T,H,W]

        # compute frame differences: [t+1] - [t]
        diffs = [gray[:, i + 1] - gray[:, i] for i in range(T - 1)]
        motion_diff = torch.stack(diffs, dim=1)  # [B,T-1,H,W]

        # convert to attention weights
        attention = torch.sigmoid(self.a * motion_diff.abs() + self.b)

        # expand to match visual features
        motion_features = attention.unsqueeze(2).expand(B, T - 1, 3, H, W)

        return motion_features, attention, None


class MotionFusion(nn.Module):
    """
    Fuse visual features with motion attention
    Support variable number of frames and handle edge cases gracefully
    """

    def __init__(self):
        super().__init__()

    def forward(self, visual, motion):
        """
        Args:
            visual: [B,T,H,W] - per-frame visual logits/features
            motion: [B,T-1,H,W] - motion attention between consecutive frames
        Returns:
            [B,T,H,W] - motion-enhanced features
        """
        if visual.ndim != 4 or motion.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got visual={visual.shape}, motion={motion.shape}")

        B, T, H, W = visual.shape
        if motion.shape[0] != B or motion.shape[2] != H or motion.shape[3] != W:
            raise ValueError(
                f"Shape mismatch: visual={visual.shape}, motion={motion.shape} "
                "(B/H/W must match)"
            )
        if T > 1 and motion.shape[1] != (T - 1):
            raise ValueError(
                f"Temporal mismatch: need exactly {T-1} motion maps for {T} frames, "
                f"got {motion.shape[1]}"
            )

        if T == 1:
            return visual

        first_frame = visual[:, :1]
        motion_enhanced = visual[:, 1:] * motion
        return torch.cat([first_frame, motion_enhanced], dim=1)

class TrackNet(nn.Module):
    """
    Motion-Enhanced U-Net for Sports Object Tracking

    Architecture:
    Input [B,num_frames*3,H,W] -> Encoder (4 blocks) -> Bottleneck -> Decoder (3 blocks) -> Output [B,num_frames,H,W]
                    -> Motion Analysis -----------------------> Motion Fusion -------^
    """

    def __init__(self, num_frames=3):
        super().__init__()
        # input channels = num_frames * 3 (RGB for each frame)
        self.num_frames = num_frames
        in_channels = num_frames * 3

        # motion processing modules
        self.motion_prompt = MotionPrompt()
        self.motion_fusion = MotionFusion()

        # encoder path: progressive downsampling
        self.enc1 = self._conv_block(in_channels, 64, 2)  # in_channels->64
        self.enc2 = self._conv_block(64, 128, 2)  # 64->128
        self.enc3 = self._conv_block(128, 256, 3)  # 128->256
        self.enc4 = self._conv_block(256, 512, 3)  # 256->512 (bottleneck)

        # decoder path: progressive upsampling with skip connections
        self.dec1 = self._conv_block(768, 256, 3)  # 512+256->256
        self.dec2 = self._conv_block(384, 128, 2)  # 256+128->128
        self.dec3 = self._conv_block(192, 64, 2)  # 128+64->64

        # utility layers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.output = nn.Conv2d(64, num_frames, 1)

    def _conv_block(self, in_ch, out_ch, n_layers):
        """Create convolution block with batch norm and ReLU"""
        layers = []
        for i in range(n_layers):
            ch_in = in_ch if i == 0 else out_ch
            layers.extend([
                nn.Conv2d(ch_in, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args: x [B,num_frames*3,288,512] - concatenated frames
        Returns: [B,num_frames,288,512] - probability heatmaps for object detection
        """
        B = x.size(0)

        # extract motion attention from input frames
        _, motion_mask, _ = self.motion_prompt(x.view(B, self.num_frames, 3, 288, 512)) # [B, num_frames-1, 288, 512]

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

        # generate base output and enhance with motion
        visual_output = self.output(d3)  # [B,num_frames,288,512]
        enhanced_output = self.motion_fusion(visual_output, motion_mask)

        return torch.sigmoid(enhanced_output)


def gaussian_heatmap(size, center, sigma=5):
    """Generate 2D Gaussian heatmap for ground truth labels"""
    H, W = size
    x, y = center
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    return torch.exp(-((X.float() - x) ** 2 + (Y.float() - y) ** 2) / (2 * sigma ** 2))


if __name__ == "__main__":
    # model initialization and testing
    num_frames = 5
    model = TrackNet(num_frames=num_frames)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TrackNet initialized with {total_params:,} parameters")

    # forward pass test
    test_input = torch.randn(2, num_frames * 3, 288, 512)
    test_output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("✓ TrackNet ready for training!")
