# ml_pipeline/model.py
import torch
import torch.nn as nn

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5) if dropout else nn.Identity()

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.activation(self.batch_norm(self.conv(x)))
        return self.dropout(x)

class StemExtractorUNet(nn.Module):
    def __init__(self, num_stems=4):
        super().__init__()
        self.num_stems = num_stems
        
        self.enc1 = UNetEncoderBlock(2, 16)
        self.enc2 = UNetEncoderBlock(16, 32)
        self.enc3 = UNetEncoderBlock(32, 64)
        self.enc4 = UNetEncoderBlock(64, 128)
        self.enc5 = UNetEncoderBlock(128, 256)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.dec5 = UNetDecoderBlock(512, 128, 128, dropout=True)
        self.dec4 = UNetDecoderBlock(128, 64, 64, dropout=True)
        self.dec3 = UNetDecoderBlock(64, 32, 32)
        self.dec2 = UNetDecoderBlock(32, 16, 16)
        
        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        # staple the raw input audio to the final layer to preserve maximum detail
        self.final_conv = nn.Conv2d(16 + 2, num_stems * 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)

        d5 = self.dec5(b, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)

        out_up = self.final_up(d2)
        out_cat = torch.cat([out_up, x], dim=1) 
        out = self.final_conv(out_cat)
        
        masks = self.sigmoid(out)
        
        batch_size, _, freq_bins, time_frames = masks.shape
        masks = masks.view(batch_size, self.num_stems, 2, freq_bins, time_frames)
        
        return masks

