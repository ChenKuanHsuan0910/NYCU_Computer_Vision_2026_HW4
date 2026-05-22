"""
PromptIR-based image restoration model with NAFNet backbone (v2).

Key improvements over v1:
- NAFBlock as core building unit (from NAFNet, ECCV 2022)
  * SimpleGate activation (splits channels and multiplies) replaces GELU/ReLU
  * Simplified Channel Attention (SCA) via AdaptiveAvgPool + 1x1 conv
  * LayerNorm2d for stable AMP training
  * Learnable residual scaling (beta/gamma init=1e-3) for stable early training
- Global Residual Learning: predict residual correction, output = residual + input
- Strided conv downsampling (better gradient flow vs MaxPool)
- Deeper bottleneck (4x NAFBlock) and num_blocks=2 per enc/dec stage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Split channels in half and multiply — replaces non-linear activations."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet block (ECCV 2022). Two residual branches:
    1. Attention: LayerNorm → expand → depthwise → SimpleGate → SCA → project
    2. FFN: LayerNorm → expand → SimpleGate → project
    Learnable per-block scaling (beta, gamma) for stable early training.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch = c * dw_expand
        # Branch 1: depthwise attention
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)
        # Branch 2: FFN
        ffn_ch = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        inp = inp + x * self.beta

        x = self.norm2(inp)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return inp + x * self.gamma


class SEBlock(nn.Module):  # kept for reference, not used in main model
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, ks, padding=ks//2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, ks, padding=ks//2)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch)
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        res = x if self.res_conv is None else self.res_conv(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.se(x)
        return x + res


class Down(nn.Module):
    """Encoder stage: project channels → NAFBlocks → strided conv downsample."""
    def __init__(self, in_ch, out_ch, num_blocks=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            *[NAFBlock(out_ch) for _ in range(num_blocks)],
        )
        self.pool = nn.Conv2d(out_ch, out_ch, 2, stride=2)  # strided conv

    def forward(self, x):
        y = self.conv(x)
        return self.pool(y), y   # (downsampled, skip)


class Up(nn.Module):
    """Decoder stage: transposed conv upsample → concat skip → NAFBlocks."""
    def __init__(self, in_ch, out_ch, num_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        # after concat: out_ch (upsampled) + in_ch (skip) = in_ch + out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 1),
            *[NAFBlock(out_ch) for _ in range(num_blocks)],
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PromptIR(nn.Module):
    """
    PromptIR-inspired restoration model with NAFNet backbone (v2).

    Architecture:
      inp_proj  : 3 → c   (3×3 conv + NAFBlock)
      down1     : c → 2c  (2× NAFBlocks + strided conv)
      down2     : 2c → 4c
      down3     : 4c → 8c
      center    : 8c      (4× NAFBlocks, deep bottleneck)
      up3       : 8c → 4c (skip from down3, 2× NAFBlocks)
      up2       : 4c → 2c
      up1       : 2c → c
      out_proj  : c → 3   (3×3 conv, no activation)
      output    : (out_proj + x).clamp(0,1)  [global residual learning]

    PromptIR components:
      prompt_embeddings : [num_prompts, prompt_dim]
      prompt_generator  : bottleneck → soft prompt weights
      film_projs        : prompt_vec → FiLM (scale+shift) for each decoder stage
    """

    def __init__(self, in_ch=3, base_ch=64, num_prompts=16, prompt_dim=128):
        super().__init__()
        c = base_ch

        self.inp_proj = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            NAFBlock(c),
        )

        self.down1 = Down(c, c * 2, num_blocks=2)
        self.down2 = Down(c * 2, c * 4, num_blocks=2)
        self.down3 = Down(c * 4, c * 8, num_blocks=2)

        self.center = nn.Sequential(*[NAFBlock(c * 8) for _ in range(4)])

        self.up3 = Up(c * 8, c * 4, num_blocks=2)
        self.up2 = Up(c * 4, c * 2, num_blocks=2)
        self.up1 = Up(c * 2, c, num_blocks=2)

        # No sigmoid — global residual: output = out_proj(d1) + x, clamped
        self.out_proj = nn.Conv2d(c, in_ch, 3, padding=1)

        # PromptIR: learnable prompt embeddings
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)

        # Prompt generator: bottleneck features → soft weights over prompts
        self.prompt_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 8, prompt_dim, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(prompt_dim, num_prompts),
        )

        # FiLM projections for each decoder stage
        self.film_projs = nn.ModuleList([
            nn.Linear(prompt_dim, c * 4 * 2),
            nn.Linear(prompt_dim, c * 2 * 2),
            nn.Linear(prompt_dim, c * 1 * 2),
        ])

    def forward(self, x):
        e = self.inp_proj(x)

        x1, s1 = self.down1(e)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)

        ct = self.center(x3)

        logits = self.prompt_generator(ct)
        weights = torch.softmax(logits, dim=-1)
        prompt_vec = torch.matmul(weights, self.prompt_embeddings)  # [B, prompt_dim]

        fps = [p(prompt_vec) for p in self.film_projs]

        d3 = self.up3(ct, s3)
        sc, sh = fps[0].view(fps[0].size(0), -1, 2).unbind(-1)
        d3 = d3 * (1 + sc.unsqueeze(-1).unsqueeze(-1)) + sh.unsqueeze(-1).unsqueeze(-1)

        d2 = self.up2(d3, s2)
        sc, sh = fps[1].view(fps[1].size(0), -1, 2).unbind(-1)
        d2 = d2 * (1 + sc.unsqueeze(-1).unsqueeze(-1)) + sh.unsqueeze(-1).unsqueeze(-1)

        d1 = self.up1(d2, s1)
        sc, sh = fps[2].view(fps[2].size(0), -1, 2).unbind(-1)
        d1 = d1 * (1 + sc.unsqueeze(-1).unsqueeze(-1)) + sh.unsqueeze(-1).unsqueeze(-1)

        residual = self.out_proj(d1)
        return (residual + x).clamp(0.0, 1.0)


def make_model(in_ch=3, base_ch=64, num_prompts=16, prompt_dim=128):
    return PromptIR(in_ch=in_ch, base_ch=base_ch, num_prompts=num_prompts, prompt_dim=prompt_dim)
