# =============================================================================
# utils/checkpoint.py
# Checkpoint Management — Face Anti-Spoofing Research Project
#
# Handles saving and loading of model checkpoints.
# Only the BEST model per experiment is saved (confirmed in config.py).
#
# Checkpoint file stores:
#   - Model state dict
#   - Optimizer state dict
#   - Best metric value and name
#   - Epoch number
#   - Full metrics dict at best epoch
#   - Experiment metadata (model, ft_level, identifier)
#
# Usage:
#   from utils.checkpoint import CheckpointManager
#   ckpt = CheckpointManager(model_name, ft_level, identifier, logger)
#   ckpt.save(model, optimizer, epoch, metric_value, metrics_dict)
#   ckpt.load(model, optimizer)  # loads best checkpoint in-place
# =============================================================================

import os
import sys
import torch
from typing import Optional, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class CheckpointManager:
    """
    Manages saving and loading of the best model checkpoint
    for a single experiment (model × ft_level × identifier).

    Only one checkpoint is kept per experiment — the best one.
    Overwrites previous best if new metric is better.

    Args:
        model_name : Model key from config.MODELS
        ft_level   : Fine-tuning level ('L1', 'L2', 'L3')
        identifier : Dataset name or Protocol B config name
        logger     : Logger instance from utils.logger
        metric_name: Metric used for best model selection
                     ('EER' or 'HTER') — lower is better
    """

    def __init__(
        self,
        model_name  : str,
        ft_level    : str,
        identifier  : str,
        logger,
        metric_name : str = None,
    ):
        self.model_name  = model_name
        self.ft_level    = ft_level
        self.identifier  = identifier
        self.logger      = logger
        self.metric_name = metric_name or cfg.PROTOCOL_A_PRIMARY_METRIC

        # Derive checkpoint path from config helper
        self.checkpoint_path = cfg.get_checkpoint_path(
            model_name, ft_level, identifier
        )

        # Track best metric seen so far (lower is better for EER/HTER)
        self.best_metric = float("inf")
        self.best_epoch  = -1

        self.logger.info(
            f"CheckpointManager initialized | "
            f"metric={self.metric_name} | "
            f"path={self.checkpoint_path}"
        )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(
        self,
        model        : torch.nn.Module,
        optimizer    : torch.optim.Optimizer,
        epoch        : int,
        metric_value : float,
        metrics_dict : Dict[str, float],
    ) -> bool:
        """
        Save checkpoint if metric_value is better than current best.

        Args:
            model        : PyTorch model
            optimizer    : Optimizer (saved for potential resuming)
            epoch        : Current epoch number (1-based)
            metric_value : Value of primary metric (lower = better)
            metrics_dict : Full metrics dict from compute_all_metrics

        Returns:
            bool: True if checkpoint was saved (new best), False otherwise
        """
        if metric_value >= self.best_metric:
            return False

        # New best — save checkpoint
        self.best_metric = metric_value
        self.best_epoch  = epoch

        checkpoint = {
            # Model and optimizer state
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            # Epoch and metric info
            "epoch"               : epoch,
            "best_metric_name"    : self.metric_name,
            "best_metric_value"   : metric_value,
            "metrics_dict"        : metrics_dict,

            # Experiment metadata
            "model_name"          : self.model_name,
            "ft_level"            : self.ft_level,
            "identifier"          : self.identifier,

            # Config snapshot for reproducibility
            "config_snapshot"     : {
                "NUM_EPOCHS"       : cfg.NUM_EPOCHS,
                "BATCH_SIZE"       : cfg.BATCH_SIZE,
                "LEARNING_RATE"    : cfg.LEARNING_RATE,
                "BACKBONE_LR"      : cfg.BACKBONE_LR,
                "CLASSIFIER_LR"    : cfg.CLASSIFIER_LR,
                "L2_FREEZE_RATIO"  : cfg.L2_FREEZE_RATIO,
                "IMAGE_SIZE"       : cfg.IMAGE_SIZE,
            },
        }

        torch.save(checkpoint, self.checkpoint_path)

        self.logger.info(
            f"Checkpoint SAVED | "
            f"epoch={epoch} | "
            f"{self.metric_name}={metric_value:.4f} "
            f"(prev best={self.best_metric if self.best_metric != metric_value else float('inf'):.4f})"
        )
        return True

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load(
        self,
        model     : torch.nn.Module,
        optimizer : Optional[torch.optim.Optimizer] = None,
        device    : Optional[torch.device] = None,
    ) -> Dict:
        """
        Load the best checkpoint into model (and optionally optimizer).

        Loads in-place — model weights are updated directly.

        Args:
            model     : PyTorch model to load weights into
            optimizer : Optional optimizer to restore state
            device    : Device to map checkpoint tensors to
                        If None, uses config.DEVICE

        Returns:
            Dict: The full checkpoint dict (for accessing metadata)

        Raises:
            FileNotFoundError: If checkpoint file does not exist
        """
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"  Has this experiment been trained? "
                f"Run Phase 3 or Phase 4 first."
            )

        map_device = device or torch.device(cfg.DEVICE)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=map_device,
            weights_only=False,
        )

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Optionally load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore internal tracking state
        self.best_metric = checkpoint.get("best_metric_value", float("inf"))
        self.best_epoch  = checkpoint.get("epoch", -1)

        self.logger.info(
            f"Checkpoint LOADED | "
            f"epoch={self.best_epoch} | "
            f"{checkpoint.get('best_metric_name', self.metric_name)}"
            f"={self.best_metric:.4f} | "
            f"path={self.checkpoint_path}"
        )

        return checkpoint

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Return True if a checkpoint file exists for this experiment."""
        return os.path.isfile(self.checkpoint_path)

    def get_best_metric(self) -> float:
        """Return best metric value seen so far."""
        return self.best_metric

    def get_best_epoch(self) -> int:
        """Return epoch number of best checkpoint."""
        return self.best_epoch

    def get_saved_metrics(self) -> Optional[Dict]:
        """
        Load and return the metrics dict from the saved checkpoint
        without modifying the model.

        Returns:
            Dict of metrics if checkpoint exists, None otherwise.
        """
        if not self.exists():
            return None

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        return checkpoint.get("metrics_dict", {})

    def get_checkpoint_info(self) -> Optional[Dict]:
        """
        Return checkpoint metadata without loading model weights.
        Useful for Phase 7 result compilation.

        Returns:
            Dict with epoch, metric, metrics_dict, config_snapshot
            or None if checkpoint does not exist.
        """
        if not self.exists():
            return None

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        return {
            "model_name"      : checkpoint.get("model_name"),
            "ft_level"        : checkpoint.get("ft_level"),
            "identifier"      : checkpoint.get("identifier"),
            "epoch"           : checkpoint.get("epoch"),
            "best_metric_name": checkpoint.get("best_metric_name"),
            "best_metric_value": checkpoint.get("best_metric_value"),
            "metrics_dict"    : checkpoint.get("metrics_dict", {}),
            "config_snapshot" : checkpoint.get("config_snapshot", {}),
        }


# =============================================================================
# STANDALONE HELPER FUNCTIONS
# Used when you need checkpoint operations outside the class context
# =============================================================================

def load_best_model(
    model      : torch.nn.Module,
    model_name : str,
    ft_level   : str,
    identifier : str,
    device     : Optional[torch.device] = None,
    logger     = None,
) -> Dict:
    """
    Convenience function to load best model without instantiating
    CheckpointManager explicitly.

    Used by Phase 6 (visualization) to load checkpoints from Phase 3.

    Args:
        model      : PyTorch model to load weights into
        model_name : Model key
        ft_level   : Fine-tuning level
        identifier : Dataset or config name
        device     : Target device
        logger     : Optional logger

    Returns:
        Dict: Full checkpoint dict
    """
    from utils.logger import get_logger
    _logger = logger or get_logger("checkpoint_loader")

    manager = CheckpointManager(
        model_name  = model_name,
        ft_level    = ft_level,
        identifier  = identifier,
        logger      = _logger,
    )
    return manager.load(model, device=device)


def checkpoint_exists(
    model_name : str,
    ft_level   : str,
    identifier : str,
) -> bool:
    """
    Check if a checkpoint exists for given experiment without
    instantiating a full CheckpointManager.

    Used by resume logic in Phase 3, 4, 5.

    Args:
        model_name : Model key
        ft_level   : Fine-tuning level
        identifier : Dataset or config name

    Returns:
        bool: True if checkpoint file exists
    """
    path = cfg.get_checkpoint_path(model_name, ft_level, identifier)
    return os.path.isfile(path)


# =============================================================================
# SELF-TEST
# Run directly: python utils/checkpoint.py
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import torch.nn as nn
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import get_logger

    print("=" * 65)
    print("CHECKPOINT SELF-TEST")
    print("=" * 65)

    logger = get_logger("test_checkpoint")

    # Simple test model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)

    device = torch.device("cpu")

    # Test 1: CheckpointManager initialization
    print("\nTEST 1: CheckpointManager initialization")
    manager = CheckpointManager(
        model_name  = "efficientnet_b0",
        ft_level    = "L2",
        identifier  = "TEST_DATASET",
        logger      = logger,
        metric_name = "EER",
    )
    print(f"  Checkpoint path : {manager.checkpoint_path}")
    print(f"  Best metric     : {manager.best_metric}")
    print("  PASS")

    # Test 2: Save — first save should always succeed
    print("\nTEST 2: First save (should save)")
    model     = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fake_metrics = {
        "EER": 0.15, "HTER": 0.15, "APCER": 0.14,
        "BPCER": 0.16, "ACER": 0.15, "AUC": 0.91,
        "Accuracy": 0.85, "Inference_ms": 2.3,
    }
    saved = manager.save(model, optimizer, epoch=5,
                         metric_value=0.15, metrics_dict=fake_metrics)
    assert saved is True, "First save should return True"
    assert manager.exists(), "Checkpoint file should exist after save"
    assert manager.best_epoch == 5
    assert manager.best_metric == 0.15
    print(f"  Saved           : {saved}")
    print(f"  Best epoch      : {manager.best_epoch}")
    print(f"  Best metric     : {manager.best_metric}")
    print("  PASS")

    # Test 3: Save — worse metric should NOT save
    print("\nTEST 3: Save with worse metric (should NOT save)")
    saved = manager.save(model, optimizer, epoch=6,
                         metric_value=0.20, metrics_dict=fake_metrics)
    assert saved is False, "Worse metric should not save"
    assert manager.best_epoch == 5, "Best epoch should still be 5"
    print(f"  Saved           : {saved}  (expected False)")
    print(f"  Best epoch      : {manager.best_epoch}  (should still be 5)")
    print("  PASS")

    # Test 4: Save — better metric SHOULD save
    print("\nTEST 4: Save with better metric (should save)")
    better_metrics = {**fake_metrics, "EER": 0.08}
    saved = manager.save(model, optimizer, epoch=10,
                         metric_value=0.08, metrics_dict=better_metrics)
    assert saved is True, "Better metric should save"
    assert manager.best_epoch == 10
    assert manager.best_metric == 0.08
    print(f"  Saved           : {saved}")
    print(f"  Best epoch      : {manager.best_epoch}")
    print(f"  Best metric     : {manager.best_metric}")
    print("  PASS")

    # Test 5: Load checkpoint
    print("\nTEST 5: Load checkpoint")
    new_model     = TinyModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

    # Verify weights differ before load
    orig_weight = model.fc.weight.data.clone()
    new_model.fc.weight.data.fill_(999.0)  # corrupt weights

    ckpt = manager.load(new_model, new_optimizer, device=device)

    assert torch.allclose(new_model.fc.weight.data, orig_weight), \
        "Loaded weights should match saved weights"
    assert ckpt["epoch"] == 10
    assert ckpt["best_metric_value"] == 0.08
    print(f"  Loaded epoch    : {ckpt['epoch']}")
    print(f"  Loaded metric   : {ckpt['best_metric_value']}")
    print("  PASS")

    # Test 6: get_saved_metrics
    print("\nTEST 6: get_saved_metrics")
    saved_metrics = manager.get_saved_metrics()
    assert saved_metrics is not None
    assert saved_metrics["EER"] == 0.08
    print(f"  EER from disk   : {saved_metrics['EER']}")
    print("  PASS")

    # Test 7: get_checkpoint_info
    print("\nTEST 7: get_checkpoint_info")
    info = manager.get_checkpoint_info()
    assert info is not None
    assert info["model_name"] == "efficientnet_b0"
    assert info["ft_level"]   == "L2"
    assert info["epoch"]      == 10
    print(f"  Model           : {info['model_name']}")
    print(f"  FT Level        : {info['ft_level']}")
    print(f"  Epoch           : {info['epoch']}")
    print("  PASS")

    # Test 8: checkpoint_exists helper
    print("\nTEST 8: checkpoint_exists standalone helper")
    exists_yes = checkpoint_exists("efficientnet_b0", "L2", "TEST_DATASET")
    exists_no  = checkpoint_exists("efficientnet_b0", "L2", "NONEXISTENT")
    assert exists_yes is True
    assert exists_no  is False
    print(f"  Exists (real)   : {exists_yes}")
    print(f"  Exists (fake)   : {exists_no}")
    print("  PASS")

    # Test 9: load_best_model standalone
    print("\nTEST 9: load_best_model standalone helper")
    fresh_model = TinyModel()
    ckpt2 = load_best_model(
        model      = fresh_model,
        model_name = "efficientnet_b0",
        ft_level   = "L2",
        identifier = "TEST_DATASET",
        device     = device,
        logger     = logger,
    )
    assert torch.allclose(fresh_model.fc.weight.data, orig_weight)
    print(f"  Weights match   : True")
    print("  PASS")

    # Test 10: FileNotFoundError on missing checkpoint
    print("\nTEST 10: FileNotFoundError on missing checkpoint")
    missing_mgr = CheckpointManager(
        "ghostnetv2_100", "L1", "MISSING_DATASET", logger
    )
    try:
        missing_mgr.load(TinyModel())
        print("  FAIL — should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  Correctly raised FileNotFoundError")
        print("  PASS")

    # Cleanup test checkpoint
    test_ckpt_path = cfg.get_checkpoint_path("efficientnet_b0", "L2", "TEST_DATASET")
    if os.path.isfile(test_ckpt_path):
        os.remove(test_ckpt_path)
        print(f"\n  Cleaned up test checkpoint")

    print("\n" + "=" * 65)
    print("ALL 10 TESTS PASSED")
    print("=" * 65)
