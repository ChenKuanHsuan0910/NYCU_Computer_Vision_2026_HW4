import os
from PIL import Image
import numpy as np


def save_tensor_as_image(tensor, path):
    # tensor: C,H,W in [0,1]
    arr = (tensor.clip(0,1) * 255.0).astype('uint8') if isinstance(tensor, np.ndarray) else None
    if arr is None:
        # assume torch tensor
        arr = tensor.cpu().numpy()
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    img = Image.fromarray(arr)
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    img.save(path)


def pil_to_uint8_array(img):
    arr = np.array(img)
    if arr.dtype != 'uint8':
        arr = (arr * 255.0).astype('uint8')
    # convert HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return arr
