import torch
import torch.nn as nn

# Small, easy-to-read UNet-like model for MNIST (28x28 grayscale)
# The network predicts noise epsilon_theta(x_t, t). We inject a timestep
# embedding and add it to intermediate feature maps so the model is aware
# of which noise level (timestep) it is denoising.


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, channels=32, time_emb_dim=128):
        super().__init__()
        # Timestep embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.conv1 = ConvBlock(1, channels)
        self.down1 = nn.Conv2d(channels, channels * 2, 4, 2, 1)  # 28->14
        self.conv2 = ConvBlock(channels * 2, channels * 2)
        self.down2 = nn.Conv2d(channels * 2, channels * 4, 4, 2, 1)  # 14->7
        self.conv3 = ConvBlock(channels * 4, channels * 4)

        # Bottleneck
        self.mid = ConvBlock(channels * 4, channels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1)  # 7->14
        self.conv4 = ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)  # 14->28
        self.conv5 = ConvBlock(channels * 2, channels)

        # Final projection to single-channel noise prediction
        self.final = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x, t_emb):
        # x: (B,1,28,28), t_emb: (B, time_emb_dim)

        # Prepare a time embedding and broadcast it to spatial dims when needed.
        temb = self.time_mlp(t_emb)  # (B, time_emb_dim)

        # Encoder
        c1 = self.conv1(x)  # (B, C, 28,28)
        d1 = self.down1(c1)  # (B, 2C,14,14)
        c2 = self.conv2(d1)
        d2 = self.down2(c2)  # (B,4C,7,7)
        c3 = self.conv3(d2)

        # Add time embedding to bottleneck by reshaping & broadcasting
        # This lets the network modulate features depending on timestep.
        b = self.mid(c3)
        B, C, H, W = b.shape
        temb_spatial = temb.view(B, -1, 1, 1).expand(-1, -1, H, W)
        # If temb_dim != C, project or truncate by conv; here we concatenate
        # to keep things simple and explicit.
        b = torch.cat([b, temb_spatial[:, :C, :, :]], dim=1)

        # Decoder
        u2 = self.up2(b)
        # Skip connection from c2
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.conv4(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.conv5(u1)

        out = self.final(u1)
        return out


def get_model(device=torch.device('cpu')):
    # Helper to create the model and move to device
    model = SmallUNet()
    return model.to(device)


if __name__ == '__main__':
    # quick smoke test
    m = get_model()
    x = torch.randn(4, 1, 28, 28)
    t = torch.randn(4, 128)
    y = m(x, t)
    print('output', y.shape)
