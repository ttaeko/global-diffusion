"""
This is a simple one-pass UNet for the 30m -> 10m and 10m -> 2m stages.
Hypothesis: the many-to-one nature of generation at this stage is reduced 
enough to implement a low-cost U-Net if enough conditioning is used.
Taeko, 17 Aug 2026
"""
import torch
import torch.nn as nn 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(inplace = True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class OnePassUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, base_channels=48):
        super().__init__()

        c1 = base_channels          # 48
        c2 = base_channels * 2      # 96
        c3 = base_channels * 4      # 192
        c4 = base_channels * 8      # 384

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            c4, c3,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(
            c3, c2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(
            c2, c1,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(c1 + c1, c1)

        # Convert the final 48 feature channels
        # into one terrain-residual channel
        self.output = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)

if __name__ == "__main__":
    model = OnePassUNet(
        in_channels=20,
        out_channels=1,
        base_channels=48,
    )

    x = torch.randn(2, 24, 256, 256)

    with torch.no_grad():
        y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")