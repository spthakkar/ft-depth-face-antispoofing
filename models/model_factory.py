# =============================================================================
# models/model_factory.py
# Model Factory — Face Anti-Spoofing Research Project
#
# Responsibilities:
#   1. Load any of the 6 models with correct source (timm / torchvision)
#   2. Replace classifier head for binary FAS output (real=0, spoof=1)
#   3. Apply fine-tuning level (L1 / L2 / L3) with correct layer freezing
#   4. Build optimizer with correct learning rate groups per level
#   5. Report model statistics (params, GFLOPs, trainable layers)
#
# Critical implementation notes (verified during pre-coding review):
#   - ShuffleNetV2: torchvision ONLY (not in timm)
#   - MobileNetV4: BatchNorm requires batch_size >= 2 during training
#   - ConvNeXt-Femto, MobileViT-XXS: nested classifier attr = 'head.fc'
#   - L2 freeze boundary: first 70% of named parameter groups
#   - L3 differential LR: backbone=BACKBONE_LR, classifier=CLASSIFIER_LR
#
# Usage:
#   from models.model_factory import build_model, build_optimizer
#   model    = build_model("efficientnet_b0", ft_level="L2")
#   optimizer = build_optimizer(model, "efficientnet_b0", ft_level="L2")
# =============================================================================

import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# =============================================================================
# SECTION 1 — CLASSIFIER REPLACEMENT
# =============================================================================

def _get_classifier(model: nn.Module, attr_path: str) -> nn.Module:
    """
    Get classifier module by attribute path.
    Handles nested attributes like 'head.fc'.

    Args:
        model     : PyTorch model
        attr_path : Dot-separated attribute path e.g. 'head.fc' or 'classifier'

    Returns:
        nn.Module: The classifier layer
    """
    parts  = attr_path.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_classifier(model: nn.Module, attr_path: str, new_classifier: nn.Module):
    """
    Set classifier module by attribute path.
    Handles nested attributes like 'head.fc'.

    Args:
        model          : PyTorch model
        attr_path      : Dot-separated attribute path
        new_classifier : New nn.Linear to replace classifier with
    """
    parts  = attr_path.split(".")
    module = model
    # Navigate to parent of target attribute
    for part in parts[:-1]:
        module = getattr(module, part)
    # Set the final attribute
    setattr(module, parts[-1], new_classifier)


def _replace_classifier(model: nn.Module, model_name: str) -> nn.Module:
    """
    Replace the final classification layer with a binary output layer.
    Preserves the in_features of the original classifier.

    Args:
        model      : Loaded PyTorch model
        model_name : Key from config.MODELS

    Returns:
        nn.Module: Model with replaced classifier
    """
    attr_path  = cfg.MODELS[model_name]["classifier_attr"]
    old_clf    = _get_classifier(model, attr_path)

    # Get input features from old classifier
    if isinstance(old_clf, nn.Linear):
        in_features = old_clf.in_features
    elif isinstance(old_clf, nn.Sequential):
        # Some models wrap Linear in Sequential
        for layer in reversed(list(old_clf.children())):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break
    else:
        raise TypeError(
            f"Cannot determine in_features for classifier type "
            f"{type(old_clf)} in model {model_name}. "
            f"Check classifier_attr in config.MODELS."
        )

    # Create new binary classifier
    new_classifier = nn.Linear(in_features, cfg.NUM_CLASSES)

    # Initialize with Kaiming uniform (good for ReLU-based networks)
    nn.init.kaiming_uniform_(new_classifier.weight, nonlinearity="linear")
    nn.init.zeros_(new_classifier.bias)

    _set_classifier(model, attr_path, new_classifier)
    return model


# =============================================================================
# SECTION 2 — FINE-TUNING LEVEL APPLICATION
# =============================================================================

def _get_classifier_param_names(model_name: str) -> List[str]:
    """
    Return the parameter name prefixes belonging to the classifier.
    Used to distinguish classifier parameters from backbone parameters
    during layer freezing and optimizer group construction.

    Args:
        model_name : Key from config.MODELS

    Returns:
        List[str]: List of parameter name prefixes for classifier layers
    """
    attr_path = cfg.MODELS[model_name]["classifier_attr"]

    # Build all possible prefix variants for the classifier
    # e.g. 'head.fc' -> ['head.fc.weight', 'head.fc.bias']
    prefixes = [attr_path]

    # For ShuffleNetV2 torchvision: classifier is 'fc'
    # For ConvNeXt/MobileViT: classifier is 'head.fc'
    # For timm standard: classifier is 'classifier'
    return prefixes


def _is_classifier_param(param_name: str, model_name: str) -> bool:
    """
    Check if a parameter belongs to the classifier head.

    Args:
        param_name : Full parameter name e.g. 'head.fc.weight'
        model_name : Key from config.MODELS

    Returns:
        bool: True if parameter is part of classifier
    """
    prefixes = _get_classifier_param_names(model_name)
    return any(param_name.startswith(prefix) for prefix in prefixes)


def apply_ft_level(
    model      : nn.Module,
    model_name : str,
    ft_level   : str,
) -> nn.Module:
    """
    Apply fine-tuning depth level to model by freezing/unfreezing layers.

    L1 — Classifier Adaptation:
        Freeze ALL backbone parameters.
        Only classifier head is trainable.

    L2 — Partial Fine-Tuning:
        Freeze first 70% of backbone parameter groups.
        Unfreeze last 30% of backbone + classifier.

    L3 — Full Fine-Tuning:
        ALL parameters are trainable.
        Learning rate differentiation is handled in build_optimizer().

    Args:
        model      : PyTorch model with replaced classifier
        model_name : Key from config.MODELS
        ft_level   : One of 'L1', 'L2', 'L3'

    Returns:
        nn.Module: Model with correct requires_grad settings
    """
    if ft_level not in ["L1", "L2", "L3"]:
        raise ValueError(f"Invalid ft_level: '{ft_level}'. Must be L1, L2, or L3.")

    # Get all named parameters
    all_params = list(model.named_parameters())
    total      = len(all_params)

    # Separate backbone and classifier param names
    backbone_params = [
        (name, param) for name, param in all_params
        if not _is_classifier_param(name, model_name)
    ]
    n_backbone = len(backbone_params)

    if ft_level == "L1":
        # ----------------------------------------------------------------
        # L1: Freeze all backbone. Train classifier only.
        # ----------------------------------------------------------------
        for name, param in all_params:
            if _is_classifier_param(name, model_name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif ft_level == "L2":
        # ----------------------------------------------------------------
        # L2: Freeze first 70% of backbone. Unfreeze last 30% + classifier.
        # ----------------------------------------------------------------
        freeze_count = int(n_backbone * cfg.L2_FREEZE_RATIO)

        for i, (name, param) in enumerate(backbone_params):
            if i < freeze_count:
                param.requires_grad = False   # frozen
            else:
                param.requires_grad = True    # trainable

        # Classifier always trainable
        for name, param in all_params:
            if _is_classifier_param(name, model_name):
                param.requires_grad = True

    elif ft_level == "L3":
        # ----------------------------------------------------------------
        # L3: All layers trainable.
        # Differential LR is applied in build_optimizer(), not here.
        # ----------------------------------------------------------------
        for name, param in all_params:
            param.requires_grad = True

    return model


# =============================================================================
# SECTION 3 — MODEL LOADING
# =============================================================================

def build_model(
    model_name  : str,
    ft_level    : str,
    pretrained  : bool = True,
    device      : Optional[torch.device] = None,
) -> nn.Module:
    """
    Load a model, replace its classifier, and apply fine-tuning level.

    This is the primary entry point for all phases.

    Args:
        model_name : Key from config.MODELS dict
        ft_level   : One of 'L1', 'L2', 'L3'
        pretrained : If True, load ImageNet pretrained weights
        device     : Target device. If None, uses config.DEVICE

    Returns:
        nn.Module: Ready-to-train model on correct device

    Raises:
        ValueError: If model_name or ft_level is invalid
    """
    if model_name not in cfg.MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Valid options: {list(cfg.MODELS.keys())}"
        )

    model_cfg = cfg.MODELS[model_name]
    source    = model_cfg["source"]
    _device   = device or torch.device(cfg.DEVICE)

    # ------------------------------------------------------------------
    # Load model from correct source
    # ------------------------------------------------------------------
    if source == "timm":
        import timm
        model = timm.create_model(
            model_cfg["timm_name"],
            pretrained  = pretrained,
            num_classes = cfg.NUM_CLASSES,  # timm replaces head directly
        )
        # timm already replaced the classifier with num_classes output
        # BUT we still call _replace_classifier to ensure:
        #   1. Consistent initialization across all models
        #   2. Correct in_features detection for nested classifiers
        # Re-replace to ensure our controlled initialization
        model = _replace_classifier(model, model_name)

    elif source == "torchvision":
        import torchvision.models as tvm
    
        if model_name == "shufflenet_v2_x2_0":
            if pretrained:
                weights = tvm.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = tvm.shufflenet_v2_x2_0(weights=weights)
    
        elif model_name == "resnet50":
            if pretrained:
                weights = tvm.ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
            model = tvm.resnet50(weights=weights)
    
        else:
            raise ValueError(
                f"Unknown torchvision model: {model_name}"
            )
    
        model = _replace_classifier(model, model_name)

    else:
        raise ValueError(
            f"Unknown source '{source}' for model '{model_name}'. "
            f"Must be 'timm' or 'torchvision'."
        )

    # ------------------------------------------------------------------
    # Apply fine-tuning level (freeze/unfreeze layers)
    # ------------------------------------------------------------------
    model = apply_ft_level(model, model_name, ft_level)

    # ------------------------------------------------------------------
    # Move to target device
    # ------------------------------------------------------------------
    model = model.to(_device)

    return model


# =============================================================================
# SECTION 4 — OPTIMIZER CONSTRUCTION
# =============================================================================

def build_optimizer(
    model      : nn.Module,
    model_name : str,
    ft_level   : str,
) -> torch.optim.Optimizer:
    """
    Build optimizer with correct learning rate groups per fine-tuning level.

    L1 and L2:
        Single learning rate (cfg.LEARNING_RATE) for all trainable params.
        Backbone params are frozen so only classifier receives updates.

    L3:
        Differential learning rates:
        - Backbone params: cfg.BACKBONE_LR  (lower, preserves ImageNet features)
        - Classifier params: cfg.CLASSIFIER_LR  (higher, adapts new head)

    Args:
        model      : Model with correct requires_grad settings from build_model
        model_name : Key from config.MODELS
        ft_level   : One of 'L1', 'L2', 'L3'

    Returns:
        torch.optim.Optimizer: Configured Adam optimizer
    """
    if ft_level in ["L1", "L2"]:
        # Single LR for all trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            trainable_params,
            lr           = cfg.LEARNING_RATE,
            weight_decay = cfg.WEIGHT_DECAY,
        )

    elif ft_level == "L3":
        # Differential LR: separate groups for backbone and classifier
        backbone_params   = []
        classifier_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_classifier_param(name, model_name):
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []

        if backbone_params:
            param_groups.append({
                "params"      : backbone_params,
                "lr"          : cfg.BACKBONE_LR,
                "weight_decay": cfg.WEIGHT_DECAY,
                "name"        : "backbone",
            })

        if classifier_params:
            param_groups.append({
                "params"      : classifier_params,
                "lr"          : cfg.CLASSIFIER_LR,
                "weight_decay": cfg.WEIGHT_DECAY,
                "name"        : "classifier",
            })

        optimizer = torch.optim.Adam(param_groups)

    else:
        raise ValueError(f"Invalid ft_level: '{ft_level}'")

    return optimizer


# =============================================================================
# SECTION 5 — MODEL STATISTICS REPORTING
# =============================================================================

def get_model_stats(
    model      : nn.Module,
    model_name : str,
    ft_level   : str,
    device     : Optional[torch.device] = None,
) -> Dict:
    """
    Compute and return model statistics for Phase 2 reporting.

    Args:
        model      : Built model from build_model()
        model_name : Key from config.MODELS
        ft_level   : Fine-tuning level applied
        device     : Device for inference time measurement

    Returns:
        Dict with keys:
            model_name, ft_level,
            total_params_M, trainable_params_M, frozen_params_M,
            trainable_percent, GFLOPs, inference_ms_cpu, status
    """
    _device = device or torch.device(cfg.DEVICE)

    # Parameter counts
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    trainable_pct    = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    # GFLOPs estimation
    try:
        from thop import profile as thop_profile
        dummy = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(_device)
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        # thop not available — estimate from param count
        gflops = -1.0
    except Exception:
        gflops = -1.0

    # CPU inference time
    try:
        cpu_device = torch.device("cpu")
        model_cpu  = model.to(cpu_device)
        model_cpu.eval()

        dummy_cpu = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model_cpu(dummy_cpu)
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.perf_counter()
                _ = model_cpu(dummy_cpu)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

        import statistics
        inference_ms_cpu = statistics.mean(times)

        # Move back to original device
        model.to(_device)
    except Exception:
        inference_ms_cpu = -1.0

    return {
        "model_name"        : model_name,
        "ft_level"          : ft_level,
        "total_params_M"    : round(total_params     / 1e6, 4),
        "trainable_params_M": round(trainable_params / 1e6, 4),
        "frozen_params_M"   : round(frozen_params    / 1e6, 4),
        "trainable_percent" : round(trainable_pct,          2),
        "GFLOPs"            : round(gflops,                 4),
        "inference_ms_cpu"  : round(inference_ms_cpu,       4),
        "status"            : "OK",
    }


def print_model_summary(
    model      : nn.Module,
    model_name : str,
    ft_level   : str,
    logger     = None,
):
    """
    Print a human-readable model summary.
    Used by Phase 2 for visual verification before training.

    Args:
        model      : Built model
        model_name : Model key
        ft_level   : Fine-tuning level
        logger     : Optional logger; if None prints to stdout
    """
    display = cfg.MODELS[model_name]["display_name"]
    group   = cfg.MODELS[model_name]["architecture_group"]

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    lines = [
        f"{'='*60}",
        f"Model       : {display} ({model_name})",
        f"Group       : {group}",
        f"FT Level    : {ft_level}",
        f"{'─'*60}",
        f"Total params: {total/1e6:.4f}M",
        f"Trainable   : {trainable/1e6:.4f}M  ({trainable/total*100:.1f}%)",
        f"Frozen      : {frozen/1e6:.4f}M  ({frozen/total*100:.1f}%)",
        f"{'─'*60}",
    ]

    # Show which layers are trainable vs frozen
    lines.append("Layer status (first 5 frozen, first 5 trainable shown):")
    frozen_shown    = 0
    trainable_shown = 0

    for name, param in model.named_parameters():
        if not param.requires_grad and frozen_shown < 5:
            lines.append(f"  FROZEN   : {name}")
            frozen_shown += 1
        elif param.requires_grad and trainable_shown < 5:
            lines.append(f"  TRAINABLE: {name}")
            trainable_shown += 1
        if frozen_shown >= 5 and trainable_shown >= 5:
            break

    lines.append(f"{'='*60}")

    output = "\n".join(lines)
    if logger:
        for line in lines:
            logger.info(line)
    else:
        print(output)


# =============================================================================
# SELF-TEST
# Run directly: python models/model_factory.py
# =============================================================================

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import get_logger

    print("=" * 65)
    print("MODEL FACTORY SELF-TEST")
    print("=" * 65)

    logger = get_logger("test_model_factory")
    device = torch.device("cpu")

    # Test all 6 models × all 3 ft_levels
    results = []

    for model_name in cfg.MODEL_NAMES:
        display = cfg.MODELS[model_name]["display_name"]
        print(f"\n{'─'*60}")
        print(f"Testing: {display} ({model_name})")
        print(f"{'─'*60}")

        for ft_level in cfg.FT_LEVELS:
            try:
                # Build model (no pretrained weights for speed)
                model = build_model(
                    model_name = model_name,
                    ft_level   = ft_level,
                    pretrained = False,
                    device     = device,
                )

                # Forward pass with batch=2 (MobileNetV4 constraint)
                model.train()
                dummy = torch.randn(2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
                out   = model(dummy)

                assert out.shape == (2, cfg.NUM_CLASSES), \
                    f"Wrong output shape: {out.shape}"

                # Build optimizer
                optimizer = build_optimizer(model, model_name, ft_level)

                # Parameter counts
                total     = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters()
                                if p.requires_grad)
                pct = trainable / total * 100

                # Verify ft_level constraints
                if ft_level == "L1":
                    # Only classifier should be trainable
                    for name, param in model.named_parameters():
                        if not _is_classifier_param(name, model_name):
                            assert not param.requires_grad, \
                                f"L1: backbone param {name} should be frozen"

                elif ft_level == "L3":
                    # All params should be trainable
                    for name, param in model.named_parameters():
                        assert param.requires_grad, \
                            f"L3: param {name} should be trainable"

                    # Verify differential LR groups
                    assert len(optimizer.param_groups) == 2, \
                        f"L3 should have 2 param groups, got {len(optimizer.param_groups)}"
                    lrs = {pg.get("name", f"group{i}"): pg["lr"]
                           for i, pg in enumerate(optimizer.param_groups)}

                status = "PASS"
                print(
                    f"  {ft_level} | "
                    f"trainable={trainable/1e6:.3f}M/{total/1e6:.3f}M "
                    f"({pct:.1f}%) | "
                    f"output={tuple(out.shape)} | {status}"
                )
                results.append((model_name, ft_level, True))

            except Exception as e:
                print(f"  {ft_level} | FAIL — {e}")
                results.append((model_name, ft_level, False))

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    passed = sum(1 for _, _, ok in results if ok)
    total  = len(results)
    print(f"Passed: {passed}/{total}  ({passed/total*100:.0f}%)")

    if passed == total:
        print("\nALL TESTS PASSED")
    else:
        failed = [(m, ft) for m, ft, ok in results if not ok]
        print(f"\nFAILED: {failed}")

    # Print sample model summary
    print(f"\n{'='*65}")
    print("SAMPLE MODEL SUMMARY — EfficientNet-B0 / L2")
    print(f"{'='*65}")
    sample_model = build_model("efficientnet_b0", "L2", pretrained=False, device=device)
    print_model_summary(sample_model, "efficientnet_b0", "L2")

    # Verify L3 differential LR groups
    print(f"\n{'='*65}")
    print("L3 DIFFERENTIAL LR VERIFICATION — GhostNetV2")
    print(f"{'='*65}")
    l3_model = build_model("ghostnetv2_100", "L3", pretrained=False, device=device)
    l3_opt   = build_optimizer(l3_model, "ghostnetv2_100", "L3")
    for pg in l3_opt.param_groups:
        name   = pg.get("name", "unnamed")
        lr     = pg["lr"]
        n_p    = len(pg["params"])
        print(f"  Group '{name}': lr={lr}, n_params={n_p}")

    assert l3_opt.param_groups[0]["lr"] == cfg.BACKBONE_LR,   "Backbone LR wrong"
    assert l3_opt.param_groups[1]["lr"] == cfg.CLASSIFIER_LR, "Classifier LR wrong"
    print("  Differential LR: CORRECT")
    print(f"\n{'='*65}")
    print("ALL MODEL FACTORY TESTS COMPLETE")
    print(f"{'='*65}")
