import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from models.model_v1 import BaseNetV1


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def multi_output_loss(outputs, targets, aux_weights=(0.4, 0.2, 0.1)):
    main, *aux = outputs
    loss = segmentation_loss(main, targets)
    for weight, aux_logits in zip(aux_weights, aux):
        loss = loss + weight * segmentation_loss(aux_logits, targets)
    return loss


def build_change_aware_sampler(dataset, seed):
    changed = []
    empty = []
    area_score = []
    for idx, label_path in enumerate(dataset.gts):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(label_path)
        ratio = float(np.count_nonzero(label >= 128)) / label.size
        if ratio == 0.0:
            empty.append(idx)
            area_score.append(0.0)
        else:
            changed.append(idx)
            # Emphasize tiny/small changes, but cap the multiplier for stability.
            area_score.append(min(5.0, 1.0 / np.sqrt(max(ratio, 1e-4))))

    weights = np.zeros(len(dataset), dtype=np.float64)
    if empty:
        weights[empty] = 0.40 / len(empty)
    changed_scores = np.asarray([area_score[i] for i in changed], dtype=np.float64)
    if changed:
        weights[changed] = 0.60 * changed_scores / changed_scores.sum()

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return sampler, {
        "samples": len(dataset),
        "empty": len(empty),
        "changed": len(changed),
        "empty_sampling_mass": 0.40,
        "changed_sampling_mass": 0.60,
    }


@torch.no_grad()
def update_ema(ema_model, model, decay, step):
    decay = min(decay, (1.0 + step) / (10.0 + step))
    model_state = model.state_dict()
    for name, ema_value in ema_model.state_dict().items():
        source = model_state[name].detach()
        if ema_value.dtype.is_floating_point:
            ema_value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            ema_value.copy_(source)


def set_learning_rate(optimizer, epoch, iteration, batches, base_lr, backbone_mult):
    lr = base_lr * (0.1 ** (epoch // 100))
    global_iteration = epoch * batches + iteration
    if global_iteration < 200:
        lr *= 0.1 + 0.9 * (global_iteration + 1) / 200.0
    optimizer.param_groups[0]["lr"] = lr * backbone_mult
    optimizer.param_groups[1]["lr"] = lr
    return lr


def score_from_cm(cm):
    tn, fp, fn, tp = [float(x) for x in cm]
    eps = np.finfo(np.float32).eps
    total = tn + fp + fn + tp
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (total + eps)
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total**2)
    kappa = (oa - pe) / (1 - pe + eps)
    return {"Kappa": float(kappa), "IoU": float(iou), "F1": float(f1), "recall": float(recall), "precision": float(precision)}


def train_one_epoch(args, loader, model, ema_model, optimizer, scaler, epoch, global_step):
    model.train()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    started = time.time()
    for iteration, (images, targets) in enumerate(loader):
        pre = images[:, :3].cuda(non_blocking=True).float()
        post = images[:, 3:6].cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()
        lr = set_learning_rate(optimizer, epoch, iteration, len(loader), args.lr, args.backbone_lr_mult)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            outputs = model(pre, post)
            loss = multi_output_loss(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        update_ema(ema_model, model, args.ema_decay, global_step)

        losses.append(float(loss.detach()))
        pred = (torch.sigmoid(outputs[0].detach()) > 0.5).long()
        meter.update_cm(pred.cpu().numpy(), targets.long().cpu().numpy())
        if iteration % 50 == 0:
            print(
                f"epoch={epoch:03d} iter={iteration:03d}/{len(loader)} "
                f"loss={losses[-1]:.4f} lr={lr:.7f} elapsed={(time.time()-started)/60:.1f}m",
                flush=True,
            )
    return float(np.mean(losses)), meter.get_scores(), global_step


@torch.no_grad()
def evaluate(loader, model, threshold=0.5):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for images, targets in loader:
        pre = images[:, :3].cuda(non_blocking=True).float()
        post = images[:, 3:6].cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()
        logits = model(pre, post)[0]
        losses.append(float(segmentation_loss(logits, targets)))
        pred = (torch.sigmoid(logits) > threshold).long()
        meter.update_cm(pred.cpu().numpy(), targets.long().cpu().numpy())
    return float(np.mean(losses)), meter.get_scores()


@torch.no_grad()
def calibrate_threshold(loader, model, thresholds):
    model.eval()
    cms = np.zeros((len(thresholds), 4), dtype=np.int64)
    for images, targets in loader:
        pre = images[:, :3].cuda(non_blocking=True).float()
        post = images[:, 3:6].cuda(non_blocking=True).float()
        gt = targets.numpy().astype(bool)
        probs = torch.sigmoid(model(pre, post)[0]).cpu().numpy()
        for index, threshold in enumerate(thresholds):
            pred = probs > threshold
            tp = np.count_nonzero(pred & gt)
            fp = np.count_nonzero(pred & ~gt)
            fn = np.count_nonzero(~pred & gt)
            tn = pred.size - tp - fp - fn
            cms[index] += np.array([tn, fp, fn, tp], dtype=np.int64)
    rows = [{"threshold": float(t), **score_from_cm(cm)} for t, cm in zip(thresholds, cms)]
    best = max(rows, key=lambda row: row["F1"])
    return best, rows


def make_datasets(args):
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    train_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.image_size, args.image_size),
        myTransforms.RandomCropResize(int(7.0 / 224.0 * args.image_size)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor(),
    ])
    eval_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.image_size, args.image_size),
        myTransforms.ToTensor(),
    ])
    return (
        myDataLoader.Dataset("train", file_root=args.data_root, transform=train_transform),
        myDataLoader.Dataset("val", file_root=args.data_root, transform=eval_transform),
        myDataLoader.Dataset("test", file_root=args.data_root, transform=eval_transform),
    )


def main(args):
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = Path(args.output_root) / f"BCDD_V1_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "trainValLog.txt"
    if log_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to mix runs: {log_path} already exists")

    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"
    config = vars(args).copy()
    config["git_sha"] = git_sha
    config["variant"] = "V1_maxpool_logits_per_image_dice_deep_supervision_ema_change_sampler"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_data, val_data, test_data = make_datasets(args)
    sampler, sampler_info = build_change_aware_sampler(train_data, args.seed)
    (run_dir / "sampler.json").write_text(json.dumps(sampler_info, indent=2), encoding="utf-8")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=seed_worker,
    )

    model = BaseNetV1(pretrained=True).cuda()
    ema_model = copy.deepcopy(model).cuda().eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
            {"params": model.swa.parameters(), "lr": args.lr},
        ],
        betas=(0.9, 0.99), weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    global_step = 0
    best_f1 = -1.0
    checkpoint_path = run_dir / "checkpoint.pth.tar"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_f1 = checkpoint["best_f1"]

    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        if not args.resume:
            log.write(f"Parameters: {total_params}\n")
            log.write("Epoch\tTrainLoss\tValLoss\tKappa\tIoU\tF1\tR\tP\n")
        for epoch in range(start_epoch, args.epochs):
            train_loss, train_scores, global_step = train_one_epoch(
                args, train_loader, model, ema_model, optimizer, scaler, epoch, global_step
            )
            val_loss, scores = evaluate(val_loader, ema_model, threshold=0.5)
            log.write(
                f"{epoch+1}\t{train_loss:.5f}\t{val_loss:.5f}\t{scores['Kappa']:.4f}\t"
                f"{scores['IoU']:.4f}\t{scores['F1']:.4f}\t{scores['recall']:.4f}\t"
                f"{scores['precision']:.4f}\n"
            )
            log.flush()
            print(
                f"VAL epoch={epoch+1:03d} F1={scores['F1']:.4f} IoU={scores['IoU']:.4f} "
                f"K={scores['Kappa']:.4f} R={scores['recall']:.4f} P={scores['precision']:.4f}",
                flush=True,
            )
            if scores["F1"] >= best_f1:
                best_f1 = scores["F1"]
                torch.save(ema_model.state_dict(), run_dir / "best_model.pth")
                (run_dir / "best_meta.json").write_text(
                    json.dumps({"epoch": epoch + 1, "threshold": 0.5, **scores}, indent=2), encoding="utf-8"
                )
            checkpoint = {
                "epoch": epoch + 1, "global_step": global_step, "best_f1": best_f1,
                "model": model.state_dict(), "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config,
            }
            torch.save(checkpoint, checkpoint_path)

    ema_model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location="cuda"))
    thresholds = np.round(np.arange(0.20, 0.801, 0.01), 2)
    best_threshold, threshold_rows = calibrate_threshold(val_loader, ema_model, thresholds)
    fixed_loss, fixed_scores = evaluate(test_loader, ema_model, threshold=0.5)
    calibrated_loss, calibrated_scores = evaluate(
        test_loader, ema_model, threshold=best_threshold["threshold"]
    )
    final_report = {
        "validation_best_threshold": best_threshold,
        "test_threshold_0.5": {"loss": fixed_loss, **fixed_scores},
        "test_calibrated": {"loss": calibrated_loss, **calibrated_scores},
        "threshold_sweep": threshold_rows,
    }
    (run_dir / "threshold_and_test.json").write_text(
        json.dumps(final_report, indent=2), encoding="utf-8"
    )
    with (run_dir / "testLog.txt").open("w", encoding="utf-8") as log:
        log.write(json.dumps(final_report, indent=2))
    print(json.dumps(final_report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/BCDD")
    parser.add_argument("--output_root", default="/content/drive/MyDrive/Mobile-CDNet/outputs/BCDD")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
