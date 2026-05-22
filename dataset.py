import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _load_img(path):
    return Image.open(path).convert('RGB')


class RestorationDataset(Dataset):
    """
    Paired restoration dataset for rain and snow.
    Expects directory structure:
      data_root/train/degraded/  (rain-1.png, snow-1.png ...)
      data_root/train/clean/     (rain_clean-1.png, snow_clean-1.png ...)

    Auto-pairs by filename pattern: rain-i <-> rain_clean-i, snow-i <-> snow_clean-i
    """

    def __init__(self, data_root, split='train', patch_size=256, val_ratio=0.05, mode='train'):
        assert split in ['train', 'val']
        self.root = os.path.join(data_root, 'train')
        degraded_dir = os.path.join(self.root, 'degraded')
        clean_dir = os.path.join(self.root, 'clean')
        assert os.path.isdir(degraded_dir) and os.path.isdir(clean_dir), 'data not found'

        # gather pairs
        degraded_files = sorted([f for f in os.listdir(degraded_dir) if f.lower().endswith('.png')])
        pairs = []
        for fname in degraded_files:
            if fname.startswith('rain-'):
                clean_name = fname.replace('rain-', 'rain_clean-')
            elif fname.startswith('snow-'):
                clean_name = fname.replace('snow-', 'snow_clean-')
            else:
                # fallback: try same name
                clean_name = fname
            if os.path.exists(os.path.join(clean_dir, clean_name)):
                pairs.append((os.path.join(degraded_dir, fname), os.path.join(clean_dir, clean_name)))

        if len(pairs) == 0:
            raise RuntimeError('No paired images found')

        # train/val split
        random.seed(0)
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        if mode == 'train':
            self.pairs = pairs[n_val:]
        else:
            self.pairs = pairs[:n_val]

        self.patch_size = patch_size
        self.mode = mode

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, target):
        # random crop
        if self.patch_size is not None:
            w, h = img.size
            if w >= self.patch_size and h >= self.patch_size:
                left = random.randint(0, w - self.patch_size)
                top = random.randint(0, h - self.patch_size)
                img = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                target = target.crop((left, top, left + self.patch_size, top + self.patch_size))
        # flips
        if random.random() < 0.5:
            img = TF.hflip(img)
            target = TF.hflip(target)
        if random.random() < 0.5:
            img = TF.vflip(img)
            target = TF.vflip(target)
        # random rotate 0/90/180/270
        k = random.randint(0, 3)
        if k > 0:
            img = img.rotate(90 * k, expand=False)
            target = target.rotate(90 * k, expand=False)
        return img, target

    def __getitem__(self, idx):
        degraded_path, clean_path = self.pairs[idx]
        img = _load_img(degraded_path)
        target = _load_img(clean_path)
        if self.mode == 'train':
            img, target = self._augment(img, target)
        # convert to tensor [0,1]
        img = TF.to_tensor(img)
        target = TF.to_tensor(target)
        return img, target


class TestDataset(Dataset):
    """Loads test/degraded/0.png .. 99.png and returns (image_tensor, filename)
    Maintains original sizes by NOT cropping/resizing.
    """

    def __init__(self, data_root):
        self.test_dir = os.path.join(data_root, 'test', 'degraded')
        assert os.path.isdir(self.test_dir), 'test degraded dir not found'
        # expect files named 0.png..99.png
        files = sorted([f for f in os.listdir(self.test_dir) if f.endswith('.png')])
        # Keep numeric ordering if possible
        try:
            files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        except Exception:
            files = sorted(files)
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.test_dir, fname)
        img = _load_img(path)
        tensor = TF.to_tensor(img)
        return tensor, fname
