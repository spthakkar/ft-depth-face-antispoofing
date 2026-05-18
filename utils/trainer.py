# =============================================================================
# utils/trainer.py
# Training Engine — Face Anti-Spoofing Research Project
#
# Core training loop used by Phase 3, Phase 4, and Phase 5.
# Ties together: model, optimizer, dataloader, metrics, checkpoint, logger.
#
# Responsibilities:
#   1. Run training loop for N epochs with early stopping
#   2. Validate after every epoch using val_loader
#   3. Save best checkpoint based on primary validation metric
#   4. Evaluate on test set ONCE after training completes
#   5. Return full metrics dict and training history
#
# Design guarantees:
#   - Test set is NEVER evaluated during training (Protocol B guarantee)
#   - Early stopping monitored on validation metric only
#   - All epoch logs written via logger
#   - Gradient clipping applied for training stability
#
# Usage:
#   from utils.trainer import Trainer
#   trainer = Trainer(model, optimizer, checkpoint_mgr, logger, device)
#   result  = trainer.run(train_loader, val_loader, test_loader, model_name, ft_level, identifier)
# =============================================================================

import os
import sys
import time
import numpy as np
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.metrics import (
    compute_all_metrics,
    compute_val_metric,
    format_metrics_for_log,
)
from utils.logger import (
    log_epoch_summary,
    log_experiment_start,
    log_experiment_end,
    ProgressLogger,
    ExperimentTimer,
)
from utils.checkpoint import CheckpointManager


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = cfg.RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """
    Core training engine for all FAS experiments.

    Handles:
      - One complete training run (N epochs with early stopping)
      - Per-epoch validation and checkpoint saving
      - Single post-training test evaluation
      - Full metrics collection and history logging

    Args:
        model           : PyTorch model from model_factory.build_model()
        optimizer       : Optimizer from model_factory.build_optimizer()
        checkpoint_mgr  : CheckpointManager instance
        logger          : Logger from utils.logger.get_logger()
        device          : torch.device to run training on
        loss_fn         : Loss function (default: CrossEntropyLoss)
        max_grad_norm   : Gradient clipping norm (0 = disabled)
    """

    def __init__(
        self,
        model          : nn.Module,
        optimizer      : torch.optim.Optimizer,
        checkpoint_mgr : CheckpointManager,
        logger,
        device         : torch.device,
        loss_fn        : nn.Module = None,
        max_grad_norm  : float = 1.0,
    ):
        self.model          = model
        self.optimizer      = optimizer
        self.checkpoint_mgr = checkpoint_mgr
        self.logger         = logger
        self.device         = device
        self.max_grad_norm  = max_grad_norm

        # Loss function — CrossEntropy for binary FAS
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(device)

        # Training history — stored per epoch for Phase 7 reporting
        self.history = {
            "train_loss" : [],
            "val_loss"   : [],
            "val_metric" : [],   # primary validation metric per epoch
        }

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def run(
        self,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        test_loader  : DataLoader,
        model_name   : str,
        ft_level     : str,
        identifier   : str,
        val_metric   : str = None,
    ) -> Dict:
        """
        Run complete training experiment:
          train N epochs -> validate each epoch -> test once at end.

        Args:
            train_loader : DataLoader for training set
            val_loader   : DataLoader for validation set
            test_loader  : DataLoader for test set (evaluated ONCE at end)
            model_name   : Model key for logging
            ft_level     : Fine-tuning level for logging
            identifier   : Dataset or config name for logging
            val_metric   : Primary metric for checkpoint selection
                           ('EER' for Protocol A, 'HTER' for Protocol B)
                           If None, uses config defaults.

        Returns:
            Dict containing:
                metrics      : Final test metrics dict
                best_epoch   : Epoch with best validation performance
                total_epochs : Total epochs trained
                history      : Per-epoch training history
        """
        set_seed()

        _val_metric = val_metric or cfg.PROTOCOL_A_PRIMARY_METRIC

        # Log experiment start banner
        log_experiment_start(
            self.logger,
            model      = model_name,
            ft_level   = ft_level,
            identifier = identifier,
            extra_info = {
                "Epochs"      : cfg.NUM_EPOCHS,
                "Batch size"  : cfg.BATCH_SIZE,
                "Val metric"  : _val_metric,
                "Device"      : str(self.device),
                "Train size"  : len(train_loader.dataset),
                "Val size"    : len(val_loader.dataset),
                "Test size"   : len(test_loader.dataset),
            },
        )

        # Reset history for this run
        self.history = {
            "train_loss" : [],
            "val_loss"   : [],
            "val_metric" : [],
        }

        best_epoch          = 0
        early_stop_counter  = 0
        total_epochs_run    = 0

        # ------------------------------------------------------------------
        # TRAINING LOOP
        # ------------------------------------------------------------------
        with ExperimentTimer(self.logger, f"Full training — {identifier}"):

            for epoch in range(1, cfg.NUM_EPOCHS + 1):
                total_epochs_run = epoch

                # ---------- TRAIN ----------
                train_loss = self._train_one_epoch(epoch)
                self.history["train_loss"].append(train_loss)

                # ---------- VALIDATE ----------
                val_loss, val_score = self._validate_one_epoch(
                    val_loader, epoch, _val_metric
                )
                self.history["val_loss"].append(val_loss)
                self.history["val_metric"].append(val_score)

                # ---------- CHECKPOINT ----------
                is_best = self.checkpoint_mgr.save(
                    model        = self.model,
                    optimizer    = self.optimizer,
                    epoch        = epoch,
                    metric_value = val_score,
                    metrics_dict = {},   # full metrics saved only at test time
                )

                if is_best:
                    best_epoch         = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                # ---------- EPOCH SUMMARY ----------
                log_epoch_summary(
                    logger       = self.logger,
                    epoch        = epoch,
                    total_epochs = cfg.NUM_EPOCHS,
                    train_loss   = train_loss,
                    val_loss     = val_loss,
                    val_metrics  = {_val_metric: val_score},
                    is_best      = is_best,
                )

                # ---------- EARLY STOPPING ----------
                if early_stop_counter >= cfg.EARLY_STOPPING_PATIENCE:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs."
                    )
                    break

        # ------------------------------------------------------------------
        # TEST EVALUATION — runs exactly ONCE after training
        # ------------------------------------------------------------------
        self.logger.info("=" * 60)
        self.logger.info("LOADING BEST CHECKPOINT FOR FINAL TEST EVALUATION")
        self.logger.info(f"Best epoch: {best_epoch} | "
                         f"Val {_val_metric}: {self.checkpoint_mgr.get_best_metric():.4f}")
        self.logger.info("=" * 60)

        # Load best checkpoint back into model
        self.checkpoint_mgr.load(self.model, device=self.device)

        # Run test evaluation
        with ExperimentTimer(self.logger, "Test evaluation"):
            test_metrics = self._evaluate(test_loader, split="test")

        # Re-save checkpoint with full test metrics
        self.checkpoint_mgr.save(
            model        = self.model,
            optimizer    = self.optimizer,
            epoch        = best_epoch,
            metric_value = self.checkpoint_mgr.get_best_metric(),
            metrics_dict = test_metrics,
        )

        # Log experiment end banner
        log_experiment_end(
            self.logger,
            model      = model_name,
            ft_level   = ft_level,
            identifier = identifier,
            metrics    = test_metrics,
        )
        # Explicit memory cleanup after each experiment
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "metrics"      : test_metrics,
            "best_epoch"   : best_epoch,
            "total_epochs" : total_epochs_run,
            "history"      : self.history,
        }

    # ------------------------------------------------------------------
    # TRAINING STEP
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Run one complete training epoch.

        Args:
            epoch : Current epoch number (1-based) for progress logging

        Returns:
            float: Average training loss for this epoch
        """
        self.model.train()

        total_loss   = 0.0
        n_batches    = len(self.train_loader)
        progress     = ProgressLogger(
            self.logger,
            total_batches = n_batches,
            log_every     = max(1, n_batches // 5),  # log ~5 times per epoch
            phase         = f"Train E{epoch}",
        )

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.loss_fn(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for training stability
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

            self.optimizer.step()

            batch_loss  = loss.item()
            total_loss += batch_loss

            progress.log(batch_idx, loss=batch_loss)

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # VALIDATION STEP
    # ------------------------------------------------------------------

    def _validate_one_epoch(
        self,
        val_loader  : DataLoader,
        epoch       : int,
        val_metric  : str,
    ) -> Tuple[float, float]:
        """
        Run one validation pass and return loss + primary metric.

        Args:
            val_loader : Validation DataLoader
            epoch      : Current epoch number for logging
            val_metric : Primary metric name ('EER' or 'HTER')

        Returns:
            Tuple of (avg_val_loss, val_metric_value)
        """
        self.model.eval()

        total_loss = 0.0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss    = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                # Get spoof probability scores (softmax of class 1)
                probs  = torch.softmax(outputs, dim=1)
                scores = probs[:, 1].cpu().numpy()   # spoof probability

                all_scores.extend(scores.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss    = total_loss / max(len(val_loader), 1)
        y_true      = np.array(all_labels,  dtype=np.int32)
        y_scores    = np.array(all_scores,  dtype=np.float32)
        metric_val  = compute_val_metric(y_true, y_scores, val_metric)

        return avg_loss, metric_val

    # ------------------------------------------------------------------
    # EVALUATION (val or test)
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        loader : DataLoader,
        split  : str = "test",
    ) -> Dict:
        """
        Full evaluation pass — computes all 7 metrics.

        Args:
            loader : DataLoader (val or test)
            split  : Split name for logging ('val' or 'test')

        Returns:
            Dict: Full metrics from compute_all_metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss    = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                probs  = torch.softmax(outputs, dim=1)
                scores = probs[:, 1].cpu().numpy()

                all_scores.extend(scores.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        y_true   = np.array(all_labels, dtype=np.int32)
        y_scores = np.array(all_scores, dtype=np.float32)

        # Compute all metrics including inference time
        metrics = compute_all_metrics(
            y_true     = y_true,
            y_scores   = y_scores,
            model      = self.model,
            device     = self.device,
            image_size = cfg.IMAGE_SIZE,
        )

        avg_loss = total_loss / max(len(loader), 1)
        metrics["loss"] = round(avg_loss, 6)

        self.logger.info(
            f"[{split.upper()}] loss={avg_loss:.4f}  "
            f"{format_metrics_for_log(metrics)}"
        )

        return metrics

    # ------------------------------------------------------------------
    # PROPERTY: train_loader
    # Stored as instance attribute when run() is called
    # Private helper for _train_one_epoch
    # ------------------------------------------------------------------

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader: DataLoader):
        self._train_loader = loader


# =============================================================================
# CONVENIENCE WRAPPER
# Used by Phase 3, 4, 5 to run a complete experiment in one call
# =============================================================================

def run_experiment(
    model_name   : str,
    ft_level     : str,
    identifier   : str,
    train_loader : DataLoader,
    val_loader   : DataLoader,
    test_loader  : DataLoader,
    logger,
    device       : torch.device,
    val_metric   : str = None,
    pretrained   : bool = True,
) -> Dict:
    """
    Build model, optimizer, checkpoint manager and run a complete experiment.

    This is the primary entry point called by Phase 3 and Phase 4
    for each model × ft_level × dataset/config combination.

    Args:
        model_name   : Model key from config.MODELS
        ft_level     : Fine-tuning level ('L1', 'L2', 'L3')
        identifier   : Dataset or Protocol B config name
        train_loader : Training DataLoader
        val_loader   : Validation DataLoader
        test_loader  : Test DataLoader (evaluated once after training)
        logger       : Logger instance
        device       : Target device
        val_metric   : Validation metric for checkpoint selection
        pretrained   : Whether to load ImageNet pretrained weights

    Returns:
        Dict with keys: metrics, best_epoch, total_epochs, history
    """
    from models.model_factory import build_model, build_optimizer

    # Build model with correct FT level
    model = build_model(
        model_name = model_name,
        ft_level   = ft_level,
        pretrained = pretrained,
        device     = device,
    )

    # Build optimizer
    optimizer = build_optimizer(model, model_name, ft_level)

    # Checkpoint manager
    _val_metric    = val_metric or cfg.PROTOCOL_A_PRIMARY_METRIC
    checkpoint_mgr = CheckpointManager(
        model_name  = model_name,
        ft_level    = ft_level,
        identifier  = identifier,
        logger      = logger,
        metric_name = _val_metric,
    )

    # Build trainer
    trainer = Trainer(
        model          = model,
        optimizer      = optimizer,
        checkpoint_mgr = checkpoint_mgr,
        logger         = logger,
        device         = device,
    )

    # Set train loader (needed by _train_one_epoch via property)
    trainer.train_loader = train_loader

    # Run experiment
    result = trainer.run(
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        model_name   = model_name,
        ft_level     = ft_level,
        identifier   = identifier,
        val_metric   = _val_metric,
    )

    return result


# =============================================================================
# SELF-TEST
# Run directly: python utils/trainer.py
# Uses tiny dummy model and dummy data to verify full training loop
# =============================================================================

if __name__ == "__main__":
    import shutil
    import tempfile
    from torch.utils.data import TensorDataset

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import get_logger
    from utils.checkpoint import CheckpointManager

    print("=" * 65)
    print("TRAINER SELF-TEST")
    print("=" * 65)

    logger = get_logger("test_trainer")
    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Tiny model for fast testing
    # ------------------------------------------------------------------
    class TinyFASModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(8, 2)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    # ------------------------------------------------------------------
    # Dummy DataLoaders with realistic label distribution
    # ------------------------------------------------------------------
    def make_dummy_loader(n=100, batch_size=8, shuffle=True):
        images = torch.randn(n, 3, 32, 32)
        labels = torch.cat([
            torch.zeros(n // 2, dtype=torch.long),
            torch.ones( n // 2, dtype=torch.long),
        ])
        # Shuffle order
        idx = torch.randperm(n)
        images, labels = images[idx], labels[idx]
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, drop_last=shuffle)

    train_loader = make_dummy_loader(n=64,  batch_size=8, shuffle=True)
    val_loader   = make_dummy_loader(n=32,  batch_size=8, shuffle=False)
    test_loader  = make_dummy_loader(n=32,  batch_size=8, shuffle=False)

    # Test 1: set_seed reproducibility
    print("\nTEST 1: set_seed reproducibility")
    set_seed(42)
    t1 = torch.randn(3)
    set_seed(42)
    t2 = torch.randn(3)
    assert torch.allclose(t1, t2), "Seeds not reproducible"
    print("  PASS")

    # Test 2: Trainer initialization
    print("\nTEST 2: Trainer initialization")
    model     = TinyFASModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_mgr  = CheckpointManager(
        "efficientnet_b0", "L2", "TEST_TRAINER", logger, "EER"
    )
    trainer = Trainer(model, optimizer, ckpt_mgr, logger, device)
    trainer.train_loader = train_loader
    print("  PASS")

    # Test 3: Single training epoch
    print("\nTEST 3: Single training epoch")
    loss = trainer._train_one_epoch(epoch=1)
    assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
    assert loss > 0, f"Loss should be positive, got {loss}"
    print(f"  Train loss: {loss:.4f}")
    print("  PASS")

    # Test 4: Single validation epoch
    print("\nTEST 4: Single validation epoch")
    val_loss, val_eer = trainer._validate_one_epoch(val_loader, epoch=1, val_metric="EER")
    assert isinstance(val_loss, float)
    assert 0.0 <= val_eer <= 1.0, f"EER out of range: {val_eer}"
    print(f"  Val loss : {val_loss:.4f}")
    print(f"  Val EER  : {val_eer:.4f}")
    print("  PASS")

    # Test 5: Full evaluation
    print("\nTEST 5: Full evaluation (_evaluate)")
    metrics = trainer._evaluate(test_loader, split="test")
    required_keys = {"EER", "HTER", "APCER", "BPCER", "ACER", "AUC", "Accuracy", "Inference_ms"}
    assert required_keys.issubset(set(metrics.keys())), \
        f"Missing keys: {required_keys - set(metrics.keys())}"
    print(f"  Metrics: {format_metrics_for_log(metrics)}")
    print("  PASS")

    # Test 6: Early stopping
    print("\nTEST 6: Early stopping")
    # Temporarily set patience to 2 for fast test
    original_patience = cfg.EARLY_STOPPING_PATIENCE
    cfg.EARLY_STOPPING_PATIENCE = 2
    original_epochs   = cfg.NUM_EPOCHS
    cfg.NUM_EPOCHS    = 10

    model2     = TinyFASModel()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ckpt_mgr2  = CheckpointManager(
        "efficientnet_b0", "L1", "TEST_EARLY_STOP", logger, "EER"
    )
    trainer2 = Trainer(model2, optimizer2, ckpt_mgr2, logger, device)
    trainer2.train_loader = train_loader

    result2 = trainer2.run(
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        model_name   = "efficientnet_b0",
        ft_level     = "L1",
        identifier   = "TEST_EARLY_STOP",
        val_metric   = "EER",
    )

    assert result2["total_epochs"] <= cfg.NUM_EPOCHS, "Should have stopped early or at max"
    assert "metrics" in result2
    assert "best_epoch" in result2
    assert "history" in result2
    print(f"  Stopped at epoch  : {result2['total_epochs']}")
    print(f"  Best epoch        : {result2['best_epoch']}")
    print(f"  History length    : {len(result2['history']['train_loss'])}")

    # Restore config
    cfg.EARLY_STOPPING_PATIENCE = original_patience
    cfg.NUM_EPOCHS              = original_epochs
    print("  PASS")

    # Test 7: Checkpoint was saved
    print("\nTEST 7: Checkpoint saved by trainer")
    assert ckpt_mgr2.exists(), "Checkpoint should exist after training"
    info = ckpt_mgr2.get_checkpoint_info()
    assert info is not None
    print(f"  Checkpoint epoch  : {info['epoch']}")
    print(f"  Best metric value : {info['best_metric_value']:.4f}")
    print("  PASS")

    # Test 8: History recorded correctly
    print("\nTEST 8: Training history")
    history = result2["history"]
    n_epochs = result2["total_epochs"]
    assert len(history["train_loss"])  == n_epochs
    assert len(history["val_loss"])    == n_epochs
    assert len(history["val_metric"])  == n_epochs
    assert all(isinstance(v, float) for v in history["train_loss"])
    print(f"  train_loss  recorded: {len(history['train_loss'])} epochs")
    print(f"  val_loss    recorded: {len(history['val_loss'])} epochs")
    print(f"  val_metric  recorded: {len(history['val_metric'])} epochs")
    print("  PASS")

    # Cleanup test checkpoints
    for exp_id in ["efficientnet_b0__L2__TEST_TRAINER",
                   "efficientnet_b0__L1__TEST_EARLY_STOP"]:
        ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, exp_id)
        if os.path.isdir(ckpt_path):
            shutil.rmtree(ckpt_path, ignore_errors=True)
    print("\n  Cleaned up test checkpoints")

    print("\n" + "=" * 65)
    print("ALL 8 TESTS PASSED")
    print("=" * 65)
