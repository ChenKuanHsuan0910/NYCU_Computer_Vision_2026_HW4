import os
import numpy as np
from PIL import Image


def validate_submission(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError('pred.npz not found')
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)
    if len(keys) != 100:
        raise ValueError('pred.npz must contain 100 entries, found %d' % len(keys))
    expected = [f"{i}.png" for i in range(100)]
    for k in expected:
        if k not in keys:
            raise ValueError(f'Missing key: {k}')
        arr = data[k]
        if arr.ndim != 3:
            raise ValueError(f'Value for {k} must be 3D (C,H,W)')
        if arr.shape[0] != 3:
            raise ValueError(f'Channel must be first dim (3,H,W) for {k}')
        if arr.dtype != np.uint8:
            raise ValueError(f'Array dtype must be uint8 for {k}')
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError(f'Array values must be in [0,255] for {k}')
    print('pred.npz validation passed')


def make_pred_npz(restored_dir, out_npz):
    # restored_dir contains 0.png..99.png
    data = {}
    for i in range(100):
        fname = f"{i}.png"
        path = os.path.join(restored_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing restored image: {path}')
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)  # CHW
        data[fname] = arr
    # save as npz
    np.savez_compressed(out_npz, **data)
    print('Saved', out_npz)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='pred.npz')
    args = parser.parse_args()
    make_pred_npz(args.restored_dir, args.out)
    validate_submission(args.out)
