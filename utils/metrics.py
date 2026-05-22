import numpy as np


def psnr(img1, img2, max_val=1.0):
    """Compute PSNR between two images in range [0,1]. img shape HWC or CHW."""
    a = np.array(img1, dtype=np.float32)
    b = np.array(img2, dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)
