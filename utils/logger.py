# =============================================================================
# utils/logger.py
# Centralized Logging Utility — Face Anti-Spoofing Research Project
#
# Provides a consistent logger for every phase.
# Each phase gets its own timestamped log file under logs/
# Console output mirrors file output.
#
# Usage:
#   from utils.logger import get_logger
#   logger = get_logger("phase3_protocol_a")
#   logger.info("Starting experiment")
#   logger.warning("Something unexpected")
#   logger.error("Something failed")
# =============================================================================

import logging
import os
import sys
from datetime import datetime


def get_logger(phase_name: str, log_level: str = None) -> logging.Logger:
    """
    Create and return a logger for a given phase.

    Creates a timestamped log file under logs/ directory.
    Also mirrors output to console if LOG_TO_CONSOLE is True in config.

    Args:
        phase_name : Name of the phase, e.g. 'phase3_protocol_a'
                     Used for log file naming and logger identity.
        log_level  : Override log level. If None, uses config.LOG_LEVEL.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Import config here (not at module level) to avoid circular imports
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config as cfg
        logs_dir       = cfg.LOGS_DIR
        to_console     = cfg.LOG_TO_CONSOLE
        default_level  = cfg.LOG_LEVEL
    except Exception:
        # Fallback if config not available
        logs_dir      = os.path.join(os.path.dirname(__file__), "..", "logs")
        to_console    = True
        default_level = "INFO"

    # Resolve log level
    level_str = (log_level or default_level).upper()
    level     = getattr(logging, level_str, logging.INFO)

    # Create logs directory if it does not exist
    os.makedirs(logs_dir, exist_ok=True)

    # Unique logger name prevents duplicate handlers across imports
    logger_name = f"FAS.{phase_name}"
    logger      = logging.getLogger(logger_name)

    # If logger already has handlers, return it as-is (already configured)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate output from root logger

    # ------------------------------------------------------------------
    # Log format
    # Format: [TIMESTAMP] [LEVEL] [phase_name] message
    # Example: [2024-03-15 14:32:01] [INFO] [phase3_protocol_a] Starting experiment
    # ------------------------------------------------------------------
    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # File handler — timestamped log file per phase per run
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{phase_name}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Console handler — mirrors file output to stdout
    # ------------------------------------------------------------------
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    # Log the file path so user knows where to find detailed logs
    logger.info(f"Logger initialized — log file: {log_filepath}")

    return logger


def get_experiment_logger(
    phase_name: str,
    model: str,
    ft_level: str,
    identifier: str,
) -> logging.Logger:
    """
    Create a logger scoped to a specific experiment.
    Used inside training loops for granular per-experiment logging.

    Args:
        phase_name : Phase name e.g. 'phase3_protocol_a'
        model      : Model key e.g. 'efficientnet_b0'
        ft_level   : Fine-tuning level e.g. 'L2'
        identifier : Dataset or config name e.g. 'OULU_NPU'

    Returns:
        logging.Logger: Logger with experiment-scoped name.
    """
    experiment_name = f"{phase_name}.{model}.{ft_level}.{identifier}"
    return get_logger(experiment_name)


class ExperimentTimer:
    """
    Context manager for timing experiments.
    Logs start, end, and elapsed time automatically.

    Usage:
        with ExperimentTimer(logger, "Training epoch 5"):
            train_one_epoch(...)
    """

    def __init__(self, logger: logging.Logger, label: str = ""):
        self.logger    = logger
        self.label     = label
        self.start_time = None

    def __enter__(self):
        from datetime import datetime
        self.start_time = datetime.now()
        self.logger.info(f"START — {self.label}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from datetime import datetime
        elapsed = datetime.now() - self.start_time
        total_seconds = elapsed.total_seconds()

        hours   = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        if exc_type is not None:
            self.logger.error(
                f"FAILED — {self.label} "
                f"(elapsed: {hours:02d}h {minutes:02d}m {seconds:05.2f}s) "
                f"Error: {exc_val}"
            )
        else:
            self.logger.info(
                f"END   — {self.label} "
                f"(elapsed: {hours:02d}h {minutes:02d}m {seconds:05.2f}s)"
            )

        # Do not suppress exceptions
        return False


class ProgressLogger:
    """
    Logs training progress at regular intervals.
    Avoids flooding the log with every single batch.

    Usage:
        progress = ProgressLogger(logger, total_batches=100, log_every=10)
        for batch_idx, (x, y) in enumerate(dataloader):
            loss = train_step(x, y)
            progress.log(batch_idx, loss=loss)
    """

    def __init__(
        self,
        logger: logging.Logger,
        total_batches: int,
        log_every: int = 10,
        phase: str = "Train",
    ):
        self.logger        = logger
        self.total_batches = total_batches
        self.log_every     = log_every
        self.phase         = phase

    def log(self, batch_idx: int, **metrics):
        """
        Log progress if at a logging interval.

        Args:
            batch_idx : Current batch index (0-based)
            **metrics : Key-value pairs to log e.g. loss=0.34, acc=0.87
        """
        if (batch_idx + 1) % self.log_every == 0 or (batch_idx + 1) == self.total_batches:
            metric_str = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            self.logger.info(
                f"[{self.phase}] "
                f"Batch [{batch_idx + 1:>4d}/{self.total_batches}] "
                f"{metric_str}"
            )


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict,
    is_best: bool = False,
):
    """
    Log a clean summary line at the end of each epoch.

    Args:
        logger       : Logger instance
        epoch        : Current epoch number (1-based)
        total_epochs : Total number of epochs
        train_loss   : Average training loss
        val_loss     : Average validation loss
        val_metrics  : Dict of validation metric name -> value
        is_best      : Whether this epoch achieved a new best validation score
    """
    metrics_str = "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in val_metrics.items()
    )

    best_marker = " *** NEW BEST ***" if is_best else ""

    logger.info(
        f"Epoch [{epoch:>3d}/{total_epochs}] "
        f"train_loss={train_loss:.4f}  "
        f"val_loss={val_loss:.4f}  "
        f"{metrics_str}"
        f"{best_marker}"
    )


def log_experiment_start(
    logger: logging.Logger,
    model: str,
    ft_level: str,
    identifier: str,
    extra_info: dict = None,
):
    """
    Log a standardized experiment start banner.

    Args:
        logger     : Logger instance
        model      : Model name
        ft_level   : Fine-tuning level
        identifier : Dataset or config name
        extra_info : Optional dict of additional info to log
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"EXPERIMENT START")
    logger.info(f"  Model      : {model}")
    logger.info(f"  FT Level   : {ft_level}")
    logger.info(f"  Target     : {identifier}")
    if extra_info:
        for key, val in extra_info.items():
            logger.info(f"  {key:<10} : {val}")
    logger.info(sep)


def log_experiment_end(
    logger: logging.Logger,
    model: str,
    ft_level: str,
    identifier: str,
    metrics: dict,
):
    """
    Log a standardized experiment end banner with final metrics.

    Args:
        logger     : Logger instance
        model      : Model name
        ft_level   : Fine-tuning level
        identifier : Dataset or config name
        metrics    : Final test metrics dict
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"EXPERIMENT END")
    logger.info(f"  Model      : {model}")
    logger.info(f"  FT Level   : {ft_level}")
    logger.info(f"  Target     : {identifier}")
    logger.info(f"  RESULTS:")
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"    {key:<15} : {val:.4f}")
        else:
            logger.info(f"    {key:<15} : {val}")
    logger.info(sep)


# =============================================================================
# SELF-TEST
# Run directly: python utils/logger.py
# =============================================================================

if __name__ == "__main__":
    import time

    print("=== LOGGER SELF-TEST ===\n")

    # Test 1: Basic logger
    print("Test 1: Basic logger creation")
    logger = get_logger("test_phase")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    print()

    # Test 2: Experiment logger
    print("Test 2: Experiment-scoped logger")
    exp_logger = get_experiment_logger(
        "phase3_protocol_a", "efficientnet_b0", "L2", "OULU_NPU"
    )
    exp_logger.info("Experiment logger working")
    print()

    # Test 3: ExperimentTimer — success case
    print("Test 3: ExperimentTimer — success")
    with ExperimentTimer(logger, "Mock training step"):
        time.sleep(0.1)
    print()

    # Test 4: ExperimentTimer — failure case
    print("Test 4: ExperimentTimer — exception handling")
    try:
        with ExperimentTimer(logger, "Mock failing step"):
            time.sleep(0.05)
            raise RuntimeError("Simulated training failure")
    except RuntimeError:
        pass
    print()

    # Test 5: ProgressLogger
    print("Test 5: ProgressLogger")
    progress = ProgressLogger(logger, total_batches=25, log_every=5)
    for i in range(25):
        progress.log(i, loss=1.0 - i * 0.04, acc=0.5 + i * 0.02)
    print()

    # Test 6: Epoch summary
    print("Test 6: Epoch summary logging")
    log_epoch_summary(
        logger,
        epoch=5,
        total_epochs=30,
        train_loss=0.3421,
        val_loss=0.4123,
        val_metrics={"EER": 0.0821, "AUC": 0.9541},
        is_best=True,
    )
    print()

    # Test 7: Experiment start/end banners
    print("Test 7: Experiment banners")
    log_experiment_start(
        logger,
        model="efficientnet_b0",
        ft_level="L2",
        identifier="OULU_NPU",
        extra_info={"Batch Size": 32, "Epochs": 30},
    )
    log_experiment_end(
        logger,
        model="efficientnet_b0",
        ft_level="L2",
        identifier="OULU_NPU",
        metrics={
            "EER": 0.0421,
            "HTER": 0.0412,
            "AUC": 0.9812,
            "Accuracy": 0.9588,
        },
    )

    print("\n=== ALL LOGGER TESTS COMPLETE ===")
