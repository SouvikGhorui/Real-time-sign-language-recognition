"""Train the sign classifier on extracted WLASL landmarks.

    python train.py --subset 100                    # official WLASL train/val/test split
    python train.py --subset 100 --split signer     # test on signers never seen in training

The official split shares most signers between train and test. --split signer gives a
better idea of how the model will do on a new person, such as you on your webcam.

Writes artifacts/<run name>/best.pt and report.json.
"""
import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from slr.features import (NUM_FRAMES, Clip, augment, prepare_clip, random_indices, sample_frames,
                          to_vector, uniform_indices)
from slr.model import SignClassifier

ROOT = Path(__file__).resolve().parent
META = ROOT / "data" / "wlasl" / "meta" / "WLASL_v0.3.json"
LANDMARKS = ROOT / "data" / "wlasl" / "landmarks"


def load_samples(subset: int) -> tuple[list[dict], list[str]]:
    """Load every clip with usable landmarks. Each sample keeps its official split."""
    entries = json.loads(META.read_text(encoding="utf-8"))[:subset]
    glosses = [entry["gloss"] for entry in entries]
    samples, missing, unusable = [], 0, 0
    for label, entry in enumerate(entries):
        for instance in entry["instances"]:
            path = LANDMARKS / f"{instance['video_id']}.npz"
            if not path.exists():
                missing += 1
                continue
            with np.load(path) as data:
                clip = prepare_clip(data["raw"], float(data["aspect"]))
            if clip is None:
                unusable += 1
                continue
            samples.append({
                "clip": clip,
                "label": label,
                "split": instance["split"],
                "signer": instance["signer_id"],
                "source": (instance["url"], instance["frame_start"], instance["frame_end"]),
            })
    print(f"Loaded {len(samples)} clips ({missing} not downloaded, {unusable} without usable landmarks)")
    return samples, glosses


def signer_split(samples: list[dict], seed: int) -> None:
    """Reassign splits so that no signer appears in more than one (about 70/15/15)."""
    counts = collections.Counter(s["signer"] for s in samples)
    signers = sorted(counts)
    random.Random(seed).shuffle(signers)
    total, seen = len(samples), 0
    assignment = {}
    for signer in signers:
        fraction = seen / total
        assignment[signer] = "test" if fraction < 0.15 else "val" if fraction < 0.30 else "train"
        seen += counts[signer]
    for sample in samples:
        sample["split"] = assignment[sample["signer"]]


def remove_duplicates(samples: list[dict]) -> list[dict]:
    """Keep each identical clip once, preferring test, then val, over train."""
    rank = {"test": 0, "val": 1, "train": 2}
    kept: dict = {}
    for sample in sorted(samples, key=lambda s: rank[s["split"]]):
        kept.setdefault((sample["source"], sample["label"]), sample)
    # The same clip labelled with two different glosses is kept: it's rare and not a leak of the answer.
    removed = len(samples) - len(kept)
    if removed:
        print(f"Removed {removed} duplicate clips that would leak between splits")
    return list(kept.values())


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], train: bool, seed: int):
        self.samples = samples
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        clip: Clip = sample["clip"]
        length = len(clip.pose)
        if self.train:
            clip = augment(sample_frames(clip, random_indices(length, self.rng)), self.rng)
        else:
            clip = sample_frames(clip, uniform_indices(length))
        return torch.from_numpy(to_vector(clip)), sample["label"]


@torch.inference_mode()
def evaluate(model: nn.Module, loader, num_classes: int) -> dict:
    model.eval()
    correct1 = correct5 = total = 0
    hits = np.zeros(num_classes)
    counts = np.zeros(num_classes)
    loss_sum = 0.0
    for inputs, labels in loader:
        logits = model(inputs)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        top5 = logits.topk(min(5, num_classes), dim=1).indices
        first = top5[:, 0] == labels
        correct1 += first.sum().item()
        correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        total += len(labels)
        np.add.at(hits, labels.numpy(), first.numpy().astype(float))
        np.add.at(counts, labels.numpy(), 1)
    seen = counts > 0
    return {
        "clips": total,
        "loss": loss_sum / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top5": correct5 / max(total, 1),
        "macro_top1": float((hits[seen] / counts[seen]).mean()) if seen.any() else 0.0,
        "per_class_hits": hits.tolist(),
        "per_class_counts": counts.tolist(),
    }


def summary(metrics: dict) -> str:
    return (f"top1 {metrics['top1']:.1%} top5 {metrics['top5']:.1%} "
            f"macro {metrics['macro_top1']:.1%} loss {metrics['loss']:.3f} ({metrics['clips']} clips)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    parser.add_argument("--split", choices=["official", "signer"], default="official")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40, help="stop after this many epochs without improvement")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run folder name under artifacts/")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = ROOT / "artifacts" / (args.name or f"wlasl{args.subset}_{args.split}")
    run_dir.mkdir(parents=True, exist_ok=True)

    samples, glosses = load_samples(args.subset)
    if args.split == "signer":
        signer_split(samples, args.seed)
    samples = remove_duplicates(samples)
    groups = {name: [s for s in samples if s["split"] == name] for name in ("train", "val", "test")}
    print({name: len(group) for name, group in groups.items()})
    if not all(groups.values()):
        raise SystemExit("A split is empty, so there is not enough data to train.")

    per_class = collections.Counter(s["label"] for s in groups["train"])
    print(f"Training clips per class: min {min(per_class.get(i, 0) for i in range(len(glosses)))}, "
          f"mean {len(groups['train']) / len(glosses):.1f}")

    make_loader = lambda group, train: torch.utils.data.DataLoader(  # noqa: E731
        ClipDataset(group, train, args.seed), batch_size=args.batch_size, shuffle=train)
    train_loader = make_loader(groups["train"], True)
    val_loader = make_loader(groups["val"], False)
    test_loader = make_loader(groups["test"], False)

    model = SignClassifier(len(glosses), hidden=args.hidden, dropout=args.dropout)
    stack = torch.cat([batch for batch, _ in make_loader(groups["train"], False)])
    model.mean.copy_(stack.mean(dim=(0, 1)))
    model.std.copy_(stack.std(dim=(0, 1)).clamp_min(1e-3))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_score, best_epoch = (-1.0, 0.0), 0
    checkpoint = run_dir / "best.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(labels)

        val = evaluate(model, val_loader, len(glosses))
        score = (val["top1"], -val["loss"])
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save({
                "model": model.state_dict(), "glosses": glosses, "num_frames": NUM_FRAMES,
                "hidden": args.hidden, "dropout": args.dropout,
            }, checkpoint)
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"epoch {epoch:3d} train loss {running / len(groups['train']):.3f} | val {summary(val)}"
                  f"{' *' if improved else ''} | {time.time() - started:.0f}s", flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    # The test set is only used here, once, with the checkpoint chosen on validation.
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    val = evaluate(model, val_loader, len(glosses))
    test = evaluate(model, test_loader, len(glosses))
    print(f"\nBest epoch {best_epoch}\nval : {summary(val)}\ntest: {summary(test)}")

    per_gloss = {
        gloss: {"correct": int(h), "clips": int(c)}
        for gloss, h, c in zip(glosses, test["per_class_hits"], test["per_class_counts"])
    }
    report = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "clips": {name: len(group) for name, group in groups.items()},
        "val": {k: v for k, v in val.items() if not k.startswith("per_class")},
        "test": {k: v for k, v in test.items() if not k.startswith("per_class")},
        "test_per_gloss": per_gloss,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Saved {checkpoint} and report.json")


if __name__ == "__main__":
    main()
