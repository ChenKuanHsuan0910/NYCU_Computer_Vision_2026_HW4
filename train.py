import os
import argparse
import time
import csv
import torch
from torch.utils.data import DataLoader
from torch import optim
from dataset import RestorationDataset
from models.promptir import make_model
from utils.losses import L1Loss, CharbonnierLoss, CombinedLoss
from torch.nn.utils import clip_grad_norm_
from utils.metrics import psnr


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, optim, scaler, device, loss_fn):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optim.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            preds = model(imgs)
            loss = loss_fn(preds, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optim.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = torch.mean(torch.abs(preds - targets)).item()
            total_loss += loss * imgs.size(0)
            # compute PSNR per image
            preds_np = (preds.cpu().numpy())
            targets_np = (targets.cpu().numpy())
            for i in range(preds_np.shape[0]):
                total_psnr += psnr(targets_np[i].transpose(1,2,0), preds_np[i].transpose(1,2,0), max_val=1.0)
            n += imgs.size(0)
    return total_loss / n, total_psnr / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/run')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--loss', type=str, default='combined', choices=['l1', 'charbonnier', 'combined'])
    parser.add_argument('--base_ch', type=int, default=48)
    parser.add_argument('--num_prompts', type=int, default=16)
    parser.add_argument('--prompt_dim', type=int, default=128)
    parser.add_argument('--sched', type=str, default='cosine', choices=['none','cosine'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Linear LR warmup epochs before cosine decay')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = RestorationDataset(args.data_root, mode='train', patch_size=args.patch_size, val_ratio=args.val_ratio)
    val_ds = RestorationDataset(args.data_root, mode='val', patch_size=None, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = make_model(in_ch=3, base_ch=args.base_ch, num_prompts=args.num_prompts, prompt_dim=args.prompt_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if args.amp and torch.cuda.is_available() else None
    if args.loss == 'l1':
        loss_fn = L1Loss()
    elif args.loss == 'charbonnier':
        loss_fn = CharbonnierLoss()
    else:
        # Charbonnier + Edge + FFT frequency loss
        loss_fn = CombinedLoss(l1_weight=0.0, charbonnier_weight=1.0, edge_weight=0.05, freq_weight=0.05)

    # scheduler: linear warmup then cosine decay
    if args.sched == 'cosine':
        warmup = args.warmup_epochs
        cosine_epochs = max(1, args.epochs - warmup)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])
    else:
        scheduler = None

    start_epoch = 0
    best_psnr = 0.0
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt.get('optimizer', {}))
        start_epoch = ckpt.get('epoch', 0)
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # CSV logging
    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_psnr'])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_loss, val_psnr = validate(model, val_loader, device)
        t1 = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_psnr={val_psnr:.3f}  time={t1-t0:.1f}s")

        # step scheduler
        if scheduler is not None:
            scheduler.step()

        # save latest (cast to Python native to avoid numpy scalar issues)
        save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'epoch': int(epoch+1), 'best_psnr': float(best_psnr)},
                        os.path.join(ckpt_dir, 'latest.pt'))

        # save best
        if val_psnr > best_psnr:
            best_psnr = float(val_psnr)
            save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                             'epoch': int(epoch+1), 'best_psnr': float(best_psnr)},
                            os.path.join(ckpt_dir, 'best.pt'))

        # append csv
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, train_loss, val_loss, val_psnr])

    print('Training finished. Best val PSNR:', best_psnr)


if __name__ == '__main__':
    main()
