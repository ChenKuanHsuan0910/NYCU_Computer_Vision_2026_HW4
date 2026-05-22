import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TestDataset
from models.promptir import make_model
from utils.image_utils import save_tensor_as_image
import torchvision.transforms.functional as TF


def load_checkpoint(path, device, base_ch=48, num_prompts=16, prompt_dim=128):
    # weights_only=False because our checkpoints store Python/numpy scalars
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = make_model(in_ch=3, base_ch=base_ch, num_prompts=num_prompts, prompt_dim=prompt_dim)
    model.load_state_dict(ckpt['model'])
    return model


def tta_predict(model, img, device):
    """
    Test-Time Augmentation: average over 8 geometric transforms
    (4 rotations x 2 flips). Each transform is applied, model runs,
    then inverse transform is applied before averaging.
    Gains ~0.3-0.8 dB PSNR for free at inference time.
    """
    img = img.to(device)            # [B, C, H, W]
    accum = None
    ops = [
        (lambda x: x,                                  lambda x: x),
        (lambda x: torch.flip(x, [-1]),                lambda x: torch.flip(x, [-1])),
        (lambda x: torch.flip(x, [-2]),                lambda x: torch.flip(x, [-2])),
        (lambda x: torch.flip(x, [-1,-2]),             lambda x: torch.flip(x, [-1,-2])),
        (lambda x: x.transpose(-1,-2),                 lambda x: x.transpose(-1,-2)),
        (lambda x: torch.flip(x.transpose(-1,-2),[-1]),lambda x: torch.flip(x,[-1]).transpose(-1,-2)),
        (lambda x: torch.flip(x.transpose(-1,-2),[-2]),lambda x: torch.flip(x,[-2]).transpose(-1,-2)),
        (lambda x: torch.flip(x.transpose(-1,-2),[-1,-2]), lambda x: torch.flip(x,[-1,-2]).transpose(-1,-2)),
    ]
    with torch.no_grad():
        for aug, inv in ops:
            x_aug = aug(img)
            pred = model(x_aug)
            pred = inv(pred)
            accum = pred if accum is None else accum + pred
    return (accum / len(ops)).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/infer')
    parser.add_argument('--batch_size', type=int, default=1)  # TTA works best with batch=1
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tta', action='store_true', default=True, help='Use TTA (default: on)')
    parser.add_argument('--no_tta', dest='tta', action='store_false')
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--num_prompts', type=int, default=16)
    parser.add_argument('--prompt_dim', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_ds = TestDataset(args.data_root)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_checkpoint(args.checkpoint, device,
                            base_ch=args.base_ch,
                            num_prompts=args.num_prompts,
                            prompt_dim=args.prompt_dim)
    model = model.to(device)
    model.eval()

    out_dir = os.path.join(args.output_dir, 'restored_images')
    os.makedirs(out_dir, exist_ok=True)

    for imgs, fnames in loader:
        if args.tta:
            preds = tta_predict(model, imgs, device).cpu()
        else:
            with torch.no_grad():
                preds = model(imgs.to(device)).cpu()
        for i in range(preds.size(0)):
            p = preds[i].numpy()   # CHW float32 in [0,1]
            save_tensor_as_image(p, os.path.join(out_dir, fnames[i]))
        print(f'  {fnames}', end='\r')
    print(f'\nInference finished (TTA={args.tta}). Restored images ->', out_dir)


if __name__ == '__main__':
    main()
