"""
Training script for BERT News Classifier.

Usage:
    python src/train.py --model bert-base-uncased --lr 2e-5 --epochs 5 \
        --batch_size 32 --dropout 0.3 --max_len 256 --output_dir checkpoints/

Reproducibility:
    Seeds are set in random, numpy, torch, and transformers.
    torch.backends.cudnn.deterministic = True
    Remaining nondeterminism: CUDA atomicAdd in multi-head attention backward pass.

Experiment tracking:
    MLflow local tracking server (mlruns/ directory).
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, set_seed

from model.bert_classifier import BertNewsClassifier, LabelSmoothingCrossEntropy
from preprocessing import (
    load_and_clean_data,
    create_stratified_split,
    create_dataloaders,
    LABEL_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """
    Set seeds across all libraries for reproducibility.

    Note: CUDA atomicAdd operations in multi-head attention backward
    pass remain nondeterministic. Setting deterministic=True would
    cause significant performance degradation (~3x slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"All seeds set to {seed}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average training loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients during fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    avg_loss = total_loss / n_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset.

    Returns dict with loss, accuracy, macro F1, and per-class metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_per_class": f1_score(all_labels, all_preds, average=None),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=list(LABEL_MAP.values()), digits=4
        ),
    }
    return metrics


def train(args: argparse.Namespace) -> None:
    """Main training loop with early stopping and MLflow tracking."""
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load and preprocess data
    train_df, test_df = load_and_clean_data(cache_dir=args.data_dir)
    train_df, val_df, _ = create_stratified_split(
        train_df, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # DataLoaders with pre-tokenized caching
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=os.path.join(args.data_dir, "cached"),
    )

    # Model
    model = BertNewsClassifier(
        model_name=args.model,
        num_classes=4,
        dropout=args.dropout,
        freeze_embeddings=args.freeze_embeddings,
        use_multi_layer_head=args.multi_layer_head,
    ).to(device)

    logger.info(f"Total params: {model.get_total_params():,}")
    logger.info(f"Trainable params: {model.get_trainable_params():,}")

    # Loss function
    # L = -(1/N) * sum_i sum_c y_ic * log(p_ic) + lambda * ||theta||^2
    # Cross-entropy with optional label smoothing + L2 weight decay via AdamW
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW with decoupled weight decay (L2 regularization)
    # Important for Transformer fine-tuning stability
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # Linear warmup then linear decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    decay_scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps - warmup_steps
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
    )

    # MLflow tracking
    try:
        import mlflow
        mlflow.set_experiment("bert-news-classifier")
        mlflow.start_run(run_name=f"run_{args.model}_lr{args.lr}_drop{args.dropout}_ep{args.epochs}")
        mlflow.log_params(vars(args))
        use_mlflow = True
        logger.info("MLflow tracking enabled")
    except ImportError:
        use_mlflow = False
        logger.warning("MLflow not installed, skipping experiment tracking")

    # Training loop with early stopping
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_f1_macro: {val_metrics['f1_macro']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_accuracy": val_metrics["accuracy"],
            }, step=epoch)

        # Early stopping on val F1 macro
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0

            checkpoint_name = f"bert-news-epoch{epoch}-f1_{best_f1:.4f}.pt"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1_macro": best_f1,
                "val_loss": val_metrics["loss"],
                "args": vars(args),
            }, checkpoint_path)

            logger.info(f"Saved best checkpoint: {checkpoint_name}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Final evaluation with classification report
    logger.info("\n" + "=" * 60)
    logger.info("Final Validation Results:")
    logger.info("=" * 60)

    # Load best checkpoint
    best_checkpoint = os.path.join(args.output_dir, f"bert-news-epoch{epoch - patience_counter}-f1_{best_f1:.4f}.pt")
    if os.path.exists(best_checkpoint):
        state = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    final_metrics = evaluate(model, val_loader, criterion, device)
    logger.info(f"\n{final_metrics['classification_report']}")
    logger.info(f"\nConfusion Matrix:\n{final_metrics['confusion_matrix']}")

    if use_mlflow:
        mlflow.log_metric("best_val_f1_macro", best_f1)
        mlflow.end_run()

    logger.info(f"\nBest val F1 macro: {best_f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT News Classifier")

    # Model
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="HuggingFace model name or path")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability (tuned: 0.3 optimal)")
    parser.add_argument("--freeze_embeddings", action="store_true", default=True,
                        help="Freeze BERT embedding layers")
    parser.add_argument("--multi_layer_head", action="store_true", default=False,
                        help="Use multi-layer attention head (improves boundary F1)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0 = disabled)")

    # Training
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (searched: [1e-5, 2e-5, 3e-5, 5e-5])")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256,
                        help="Max sequence length for tokenizer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="L2 weight decay for AdamW (lambda=0.01)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of steps for linear warmup")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience (epochs)")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--num_workers", type=int, default=4)

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
