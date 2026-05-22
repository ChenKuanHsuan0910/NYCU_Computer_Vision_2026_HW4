import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        return self.criterion(pred, target)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred, target):
        # pred/target: BxCxHxW in [0,1]
        # cast kernels to match input dtype/device (important for AMP half precision)
        kx = self.kx.to(dtype=pred.dtype, device=pred.device)
        ky = self.ky.to(dtype=pred.dtype, device=pred.device)
        B,C,H,W = pred.shape
        grad_pred_x = F.conv2d(pred.view(B*C,1,H,W), kx, padding=1)
        grad_pred_y = F.conv2d(pred.view(B*C,1,H,W), ky, padding=1)
        grad_t_x = F.conv2d(target.view(B*C,1,H,W), kx, padding=1)
        grad_t_y = F.conv2d(target.view(B*C,1,H,W), ky, padding=1)
        mag_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
        mag_t = torch.sqrt(grad_t_x**2 + grad_t_y**2 + 1e-6)
        loss = torch.mean(torch.abs(mag_pred - mag_t))
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.0, charbonnier_weight=1.0, edge_weight=0.05, freq_weight=0.05):
        super().__init__()
        self.l1 = L1Loss()
        self.charb = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.freq = FrequencyLoss()
        self.l1_w = l1_weight
        self.charb_w = charbonnier_weight
        self.edge_w = edge_weight
        self.freq_w = freq_weight

    def forward(self, pred, target):
        loss = 0.0
        if self.l1_w > 0:
            loss = loss + self.l1_w * self.l1(pred, target)
        if self.charb_w > 0:
            loss = loss + self.charb_w * self.charb(pred, target)
        if self.edge_w > 0:
            loss = loss + self.edge_w * self.edge(pred, target)
        if self.freq_w > 0:
            loss = loss + self.freq_w * self.freq(pred, target)
        return loss


# Optional SSIM placeholder (not used by default)
class FrequencyLoss(nn.Module):
    """
    FFT-based frequency loss. Computes L1 on the magnitude spectrum of
    the 2D FFT. Encourages the model to recover high-frequency details
    (textures, edges) that are often lost with pure spatial losses.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # cast to float32 for FFT stability under AMP
        pred_f = pred.float()
        target_f = target.float()
        pred_fft = torch.fft.rfft2(pred_f, norm='ortho')
        target_fft = torch.fft.rfft2(target_f, norm='ortho')
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return torch.mean(torch.abs(pred_mag - target_mag))


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))
