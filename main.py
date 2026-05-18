# =============================================================================
# main.py
# Entry Point — Face Anti-Spoofing Research Project
#
# Routes command-line arguments to the correct phase module.
# Each phase runs independently and can be resumed at any time.
#
# Usage:
#   python main.py --phase 1                         # Data preparation
#   python main.py --phase 2                         # Model setup verification
#   python main.py --phase 3                         # Protocol A training
#   python main.py --phase 4                         # Protocol B training
#   python main.py --phase 5                         # Ablation studies
#   python main.py --phase 6                         # Visualization
#   python main.py --phase 7                         # Results compilation
#   python main.py --phase all                       # Run all phases in order
#
# Optional filters (Phase 3, 4, 5 only):
#   --model   mobilenetv4_conv_small                 # Run one model only
#   --ft      L2                                     # Run one FT level only
#   --dataset OULU_NPU                               # Run one dataset only (Phase 3)
#   --config  RpM_to_O                               # Run one config only (Phase 4)
#
# Examples:
#   python main.py --phase 3 --model efficientnet_b0 --ft L2
#   python main.py --phase 4 --config RpM_to_O
#   python main.py --phase 3 --dataset MSU_MFSD
# =============================================================================

import argparse
import sys
import os
import traceback
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from utils.logger import get_logger


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""

    parser = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Face Anti-Spoofing — Transfer Learning Depth Analysis",
        formatter_class = argparse.RawTextHelpFormatter,
        epilog = """
Examples:
  python main.py --phase 1
  python main.py --phase 3 --model efficientnet_b0 --ft L2
  python main.py --phase 3 --dataset OULU_NPU
  python main.py --phase 4 --config RpM_to_O
  python main.py --phase all
        """,
    )

    # Required argument
    parser.add_argument(
        "--phase",
        type    = str,
        required= True,
        choices = ["1", "2", "3", "4", "5", "6", "7", "all"],
        help    = (
            "Phase to run:\n"
            "  1   — Data preparation & face alignment\n"
            "  2   — Model setup & layer report\n"
            "  3   — Protocol A: intra-dataset training\n"
            "  4   — Protocol B: cross-dataset training\n"
            "  5   — Ablation studies\n"
            "  6   — Grad-CAM & t-SNE visualization\n"
            "  7   — Results compilation & CSV report\n"
            "  all — Run all phases sequentially"
        ),
    )

    # Optional filters
    parser.add_argument(
        "--model",
        type    = str,
        default = None,
        choices = cfg.MODEL_NAMES + [None],
        metavar = "MODEL_NAME",
        help    = (
            f"Run only this model. Options:\n"
            f"  {chr(10).join('  ' + m for m in cfg.MODEL_NAMES)}"
        ),
    )

    parser.add_argument(
        "--ft",
        type    = str,
        default = None,
        choices = cfg.FT_LEVELS + [None],
        metavar = "FT_LEVEL",
        help    = "Run only this fine-tuning level (L1, L2, or L3)",
    )

    parser.add_argument(
        "--dataset",
        type    = str,
        default = None,
        choices = list(cfg.DATASETS.keys()) + [None],
        metavar = "DATASET_NAME",
        help    = (
            "Run only this dataset (Phase 3 only). Options:\n"
            f"  {', '.join(cfg.DATASETS.keys())}"
        ),
    )

    parser.add_argument(
        "--config",
        type    = str,
        default = None,
        choices = cfg.PROTOCOL_B_CONFIG_NAMES + [None],
        metavar = "CONFIG_NAME",
        help    = (
            "Run only this Protocol B config (Phase 4 only). Options:\n"
            f"  {', '.join(cfg.PROTOCOL_B_CONFIG_NAMES)}"
        ),
    )

    parser.add_argument(
        "--no-pretrained",
        action  = "store_true",
        default = False,
        help    = "Do not load ImageNet pretrained weights (for debugging only)",
    )

    parser.add_argument(
        "--skip-validation",
        action  = "store_true",
        default = False,
        help    = "Skip config validation check at startup",
    )

    return parser


# =============================================================================
# PHASE RUNNERS
# =============================================================================

def run_phase_1(args, logger):
    """Phase 1: Data preparation and face alignment."""
    logger.info("Starting Phase 1: Data Preparation & Face Alignment")
    from phases.phase1_data_preparation import run as phase_run
    phase_run(logger=logger)


def run_phase_2(args, logger):
    """Phase 2: Model setup and layer verification."""
    logger.info("Starting Phase 2: Model Setup & Fine-Tuning Verification")
    from phases.phase2_model_setup import run as phase_run
    phase_run(
        model_filter   = args.model,
        ft_level_filter= args.ft,
        logger         = logger,
        pretrained     = not args.no_pretrained,
    )


def run_phase_3(args, logger):
    """Phase 3: Protocol A intra-dataset training."""
    logger.info("Starting Phase 3: Protocol A — Intra-Dataset Training")
    from phases.phase3_protocol_a import run as phase_run
    phase_run(
        model_filter   = args.model,
        ft_level_filter= args.ft,
        dataset_filter = args.dataset,
        logger         = logger,
        pretrained     = not args.no_pretrained,
    )


def run_phase_4(args, logger):
    """Phase 4: Protocol B cross-dataset training."""
    logger.info("Starting Phase 4: Protocol B — Cross-Dataset Training")
    from phases.phase4_protocol_b import run as phase_run
    phase_run(
        model_filter   = args.model,
        ft_level_filter= args.ft,
        config_filter  = args.config,
        logger         = logger,
        pretrained     = not args.no_pretrained,
    )


def run_phase_5(args, logger):
    """Phase 5: Ablation studies."""
    logger.info("Starting Phase 5: Ablation Studies")
    from phases.phase5_ablation import run as phase_run
    phase_run(
        model_filter   = args.model,
        ft_level_filter= args.ft,
        logger         = logger,
        pretrained     = not args.no_pretrained,
    )


def run_phase_6(args, logger):
    """Phase 6: Grad-CAM and t-SNE visualization."""
    logger.info("Starting Phase 6: Grad-CAM & t-SNE Visualization")
    from phases.phase6_visualization import run as phase_run
    phase_run(
        model_filter   = args.model,
        ft_level_filter= args.ft,
        logger         = logger,
    )


def run_phase_7(args, logger):
    """Phase 7: Results compilation and CSV report generation."""
    logger.info("Starting Phase 7: Results Compilation & CSV Reports")
    from phases.phase7_results import run as phase_run
    phase_run(logger=logger)


# Map phase number to runner function
PHASE_RUNNERS = {
    "1": run_phase_1,
    "2": run_phase_2,
    "3": run_phase_3,
    "4": run_phase_4,
    "5": run_phase_5,
    "6": run_phase_6,
    "7": run_phase_7,
}


# =============================================================================
# STARTUP BANNER
# =============================================================================

def print_startup_banner(args, logger):
    """Print a clear startup banner showing what will run."""
    sep = "=" * 65
    logger.info(sep)
    logger.info("FACE ANTI-SPOOFING — TRANSFER LEARNING DEPTH ANALYSIS")
    logger.info(sep)
    logger.info(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Device      : {cfg.DEVICE}")
    logger.info(f"  Phase       : {args.phase}")

    if args.model:
        logger.info(f"  Model filter: {args.model}")
    if args.ft:
        logger.info(f"  FT filter   : {args.ft}")
    if args.dataset:
        logger.info(f"  Dataset     : {args.dataset}")
    if args.config:
        logger.info(f"  PB config   : {args.config}")
    if args.no_pretrained:
        logger.info(f"  Pretrained  : DISABLED (debug mode)")

    logger.info(f"  Models      : {len(cfg.MODEL_NAMES)} configured")
    logger.info(f"  FT levels   : {cfg.FT_LEVELS}")
    logger.info(f"  Datasets    : {list(cfg.DATASETS.keys())}")
    logger.info(sep)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Create main logger
    phase_label = f"phase{args.phase}" if args.phase != "all" else "all_phases"
    logger      = get_logger(f"main_{phase_label}")

    # Print startup banner
    print_startup_banner(args, logger)

    # ------------------------------------------------------------------
    # Config validation (can be skipped with --skip-validation)
    # ------------------------------------------------------------------
    if not args.skip_validation:
        logger.info("Running config validation...")
        try:
            cfg.validate_config()
            logger.info("Config validation PASSED")
        except ValueError as e:
            logger.error(str(e))
            logger.error(
                "Fix the errors above then rerun.\n"
                "  Use --skip-validation to bypass (not recommended)."
            )
            sys.exit(1)
    else:
        logger.warning(
            "Config validation SKIPPED (--skip-validation flag set). "
            "Dataset paths may not exist."
        )

    # ------------------------------------------------------------------
    # Run requested phase(s)
    # ------------------------------------------------------------------
    start_time = datetime.now()

    if args.phase == "all":
        logger.info("Running ALL phases sequentially (1 → 7)")
        for phase_num in ["1", "2", "3", "4", "5", "6", "7"]:
            phase_start = datetime.now()
            logger.info(f"\n{'─'*65}")
            logger.info(f"STARTING PHASE {phase_num}")
            logger.info(f"{'─'*65}")
            try:
                PHASE_RUNNERS[phase_num](args, logger)
                elapsed = (datetime.now() - phase_start).total_seconds()
                logger.info(
                    f"Phase {phase_num} COMPLETED in "
                    f"{elapsed/60:.1f} minutes"
                )
            except Exception as e:
                logger.error(f"Phase {phase_num} FAILED: {e}")
                logger.error(traceback.format_exc())
                logger.error(
                    f"Stopping execution. Fix the error and rerun "
                    f"from phase {phase_num}."
                )
                sys.exit(1)

    else:
        # Run single phase
        runner = PHASE_RUNNERS[args.phase]
        try:
            runner(args, logger)
        except KeyboardInterrupt:
            logger.warning(
                f"\nPhase {args.phase} interrupted by user (Ctrl+C).\n"
                f"  Progress is saved. Rerun the same command to resume."
            )
            sys.exit(0)
        except Exception as e:
            logger.error(f"Phase {args.phase} FAILED: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total_elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 65)
    logger.info(
        f"DONE — Phase {args.phase} completed in "
        f"{total_elapsed/60:.1f} minutes"
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
