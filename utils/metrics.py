# =============================================================================
# utils/metrics.py
# Metrics Computation — Face Anti-Spoofing Research Project
#
# Computes all 7 required metrics from model output scores:
#   1. EER   — Equal Error Rate
#   2. HTER  — Half Total Error Rate
#   3. APCER — Attack Presentation Classification Error Rate
#   4. BPCER — Bona Fide Presentation Classification Error Rate
#   5. ACER  — Average Classification Error Rate
#   6. AUC   — Area Under ROC Curve
#   7. Accuracy
#
# Label convention (confirmed in config.py):
#   real  -> 0  (bona fide)
#   spoof -> 1  (attack / presentation attack)
#
# All functions accept:
#   y_true  : numpy array of ground truth labels (0 or 1)
#   y_scores: numpy array of spoof probability scores (float in [0,1])
#             Higher score = more likely to be spoof
#
# Usage:
#   from utils.metrics import compute_all_metrics
#   results = compute_all_metrics(y_true, y_scores)
# =============================================================================

import numpy as np
import time
import torch
from sklearn.metrics import roc_curve, auc as sklearn_auc
from typing import Dict, Tuple


# =============================================================================
# INDIVIDUAL METRIC FUNCTIONS
# =============================================================================

def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    EER is the point on the ROC curve where False Positive Rate (FPR)
    equals False Negative Rate (FNR = 1 - TPR).
    Lower EER = better model.

    Args:
        y_true   : Ground truth labels. real=0, spoof=1
        y_scores : Spoof probability scores in [0, 1]

    Returns:
        Tuple of (eer_value, eer_threshold)
        eer_value     : EER as a float in [0, 1]
        eer_threshold : Decision threshold at EER point
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find index where FPR and FNR are closest
    abs_diff = np.abs(fnr - fpr)
    eer_idx  = np.nanargmin(abs_diff)

    # EER is average of FPR and FNR at crossing point
    eer_value     = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    return eer_value, eer_threshold


def compute_hter(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute Half Total Error Rate (HTER) at a given threshold.

    HTER = (FAR + FRR) / 2
    where:
      FAR = False Acceptance Rate = FP / (FP + TN)  [real accepted as spoof... wait]
      
    Note on FAS convention:
      In FAS, "acceptance" means accepting a spoof as real (dangerous).
      FAR = spoof images classified as real / total spoof images = FNR
      FRR = real images classified as spoof / total real images = FPR

    So in terms of sklearn convention (spoof=positive=1):
      FAR = FN / (FN + TP) = missed spoofs / total spoofs = FNR
      FRR = FP / (FP + TN) = false alarms  / total reals = FPR
      HTER = (FAR + FRR) / 2 = (FNR + FPR) / 2

    Args:
        y_true    : Ground truth labels. real=0, spoof=1
        y_scores  : Spoof probability scores in [0, 1]
        threshold : Decision threshold (typically EER threshold)

    Returns:
        float: HTER value in [0, 1]
    """
    y_pred = (y_scores >= threshold).astype(int)

    total_spoof = np.sum(y_true == 1)
    total_real  = np.sum(y_true == 0)

    # FAR: spoofs incorrectly classified as real (false negatives on spoof)
    FAR = float(np.sum((y_pred == 0) & (y_true == 1)) / total_spoof) \
          if total_spoof > 0 else 0.0

    # FRR: reals incorrectly classified as spoof (false positives on real)
    FRR = float(np.sum((y_pred == 1) & (y_true == 0)) / total_real) \
          if total_real > 0 else 0.0

    return float((FAR + FRR) / 2.0)


def compute_apcer_bpcer_acer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """
    Compute APCER, BPCER, ACER per ISO/IEC 30107-3 standard.

    Definitions:
      APCER = Attack Presentation Classification Error Rate
            = proportion of attack presentations incorrectly classified as bona fide
            = FN / (FN + TP)   [missed attacks / total attacks]

      BPCER = Bona Fide Presentation Classification Error Rate
            = proportion of bona fide presentations incorrectly classified as attacks
            = FP / (FP + TN)   [false alarms / total real]

      ACER  = (APCER + BPCER) / 2

    Args:
        y_true    : Ground truth labels. real=0, spoof=1
        y_scores  : Spoof probability scores in [0, 1]
        threshold : Decision threshold

    Returns:
        Tuple of (APCER, BPCER, ACER) all in [0, 1]
    """
    y_pred = (y_scores >= threshold).astype(int)

    total_attack   = np.sum(y_true == 1)  # total spoof samples
    total_bonafide = np.sum(y_true == 0)  # total real samples

    # APCER: attacks classified as bona fide (false negatives on attack)
    APCER = float(np.sum((y_pred == 0) & (y_true == 1)) / total_attack) \
            if total_attack > 0 else 0.0

    # BPCER: bona fide classified as attack (false positives on real)
    BPCER = float(np.sum((y_pred == 1) & (y_true == 0)) / total_bonafide) \
            if total_bonafide > 0 else 0.0

    ACER = float((APCER + BPCER) / 2.0)

    return APCER, BPCER, ACER


def compute_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute Area Under the ROC Curve (AUC).

    AUC of 1.0 = perfect classifier
    AUC of 0.5 = random classifier
    Higher AUC = better model.

    Args:
        y_true   : Ground truth labels. real=0, spoof=1
        y_scores : Spoof probability scores in [0, 1]

    Returns:
        float: AUC value in [0, 1]
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    return float(sklearn_auc(fpr, tpr))


def compute_accuracy(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute classification accuracy at a given threshold.

    Args:
        y_true    : Ground truth labels. real=0, spoof=1
        y_scores  : Spoof probability scores in [0, 1]
        threshold : Decision threshold

    Returns:
        float: Accuracy in [0, 1]
    """
    y_pred = (y_scores >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


# =============================================================================
# INFERENCE TIME MEASUREMENT
# =============================================================================

def measure_inference_time(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 224,
    n_warmup: int = 10,
    n_measure: int = 100,
) -> float:
    """
    Measure average inference time per image in milliseconds.

    Uses a warm-up phase to avoid cold-start GPU overhead,
    then averages over n_measure forward passes.

    Args:
        model      : PyTorch model in eval mode
        device     : torch.device ('cuda' or 'cpu')
        image_size : Input image size (default 224)
        n_warmup   : Number of warm-up forward passes (not timed)
        n_measure  : Number of timed forward passes

    Returns:
        float: Average inference time per image in milliseconds
    """
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    # Warm-up passes — not timed
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Timed passes
    if device.type == "cuda":
        # Use CUDA events for accurate GPU timing
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        times   = []

        with torch.no_grad():
            for _ in range(n_measure):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))

        avg_ms = float(np.mean(times))

    else:
        # CPU timing using time.perf_counter
        times = []
        with torch.no_grad():
            for _ in range(n_measure):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)  # convert to ms

        avg_ms = float(np.mean(times))

    return avg_ms


# =============================================================================
# MASTER FUNCTION — computes all metrics in one call
# =============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model: torch.nn.Module = None,
    device: torch.device = None,
    image_size: int = 224,
) -> Dict[str, float]:
    """
    Compute all required metrics from model predictions.

    This is the primary function called by trainer.py, Phase 3, Phase 4,
    and Phase 5 after every evaluation pass.

    The EER threshold is used as the decision threshold for all
    threshold-dependent metrics (HTER, APCER, BPCER, ACER, Accuracy).
    This is standard practice in FAS evaluation literature.

    Args:
        y_true     : 1D numpy array of ground truth labels (0=real, 1=spoof)
        y_scores   : 1D numpy array of spoof probability scores in [0,1]
        model      : Optional — PyTorch model for inference time measurement
                     If None, Inference_ms is set to -1.0
        device     : Optional — required if model is provided
        image_size : Input image size for inference time measurement

    Returns:
        Dict with keys:
            EER, HTER, APCER, BPCER, ACER, AUC, Accuracy, Inference_ms
        All values are floats.
        EER, HTER, APCER, BPCER, ACER, Accuracy are in [0, 1].
        AUC is in [0, 1].
        Inference_ms is in milliseconds (or -1.0 if not measured).
    """
    # Input validation
    y_true   = np.asarray(y_true,   dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    assert y_true.shape   == y_scores.shape, \
        f"Shape mismatch: y_true={y_true.shape}, y_scores={y_scores.shape}"
    assert len(y_true) > 0, \
        "Empty arrays passed to compute_all_metrics"
    assert set(np.unique(y_true)).issubset({0, 1}), \
        f"y_true must contain only 0 and 1. Found: {np.unique(y_true)}"

    # Step 1: EER and threshold
    eer_value, eer_threshold = compute_eer(y_true, y_scores)

    # Step 2: HTER at EER threshold
    hter = compute_hter(y_true, y_scores, eer_threshold)

    # Step 3: APCER, BPCER, ACER at EER threshold
    apcer, bpcer, acer = compute_apcer_bpcer_acer(y_true, y_scores, eer_threshold)

    # Step 4: AUC
    auc_value = compute_auc(y_true, y_scores)

    # Step 5: Accuracy at EER threshold
    accuracy = compute_accuracy(y_true, y_scores, eer_threshold)

    # Step 6: Inference time
    if model is not None and device is not None:
        inference_ms = measure_inference_time(model, device, image_size)
    else:
        inference_ms = -1.0

    return {
        "EER"         : round(eer_value,   6),
        "HTER"        : round(hter,        6),
        "APCER"       : round(apcer,       6),
        "BPCER"       : round(bpcer,       6),
        "ACER"        : round(acer,        6),
        "AUC"         : round(auc_value,   6),
        "Accuracy"    : round(accuracy,    6),
        "Inference_ms": round(inference_ms, 4),
        "_threshold"  : round(eer_threshold, 6),  # stored for reference, not in CSV
    }


def compute_val_metric(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_name: str = "EER",
) -> float:
    """
    Compute a single metric for validation-time model selection.
    Faster than compute_all_metrics — computes only what is needed.

    Used by trainer.py during the validation loop each epoch to
    decide whether to save a new best checkpoint.

    Args:
        y_true      : Ground truth labels
        y_scores    : Spoof probability scores
        metric_name : 'EER' or 'HTER'

    Returns:
        float: Metric value. Lower is better for both EER and HTER.
    """
    y_true   = np.asarray(y_true,   dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    if metric_name == "EER":
        eer_value, _ = compute_eer(y_true, y_scores)
        return eer_value

    elif metric_name == "HTER":
        eer_value, eer_threshold = compute_eer(y_true, y_scores)
        return compute_hter(y_true, y_scores, eer_threshold)

    else:
        raise ValueError(
            f"Unsupported metric_name: '{metric_name}'. "
            f"Use 'EER' or 'HTER'."
        )


def format_metrics_for_log(metrics: Dict[str, float]) -> str:
    """
    Format a metrics dictionary into a clean single-line string for logging.

    Args:
        metrics : Dict from compute_all_metrics

    Returns:
        str: Formatted string e.g.
             "EER=4.21%  HTER=4.12%  APCER=3.95%  BPCER=4.48%  ACER=4.21%  AUC=0.9812  Acc=95.88%"
    """
    pct_keys = {"EER", "HTER", "APCER", "BPCER", "ACER", "Accuracy"}
    parts    = []

    display_names = {
        "EER"         : "EER",
        "HTER"        : "HTER",
        "APCER"       : "APCER",
        "BPCER"       : "BPCER",
        "ACER"        : "ACER",
        "AUC"         : "AUC",
        "Accuracy"    : "Acc",
        "Inference_ms": "Inf_ms",
    }

    for key in ["EER", "HTER", "APCER", "BPCER", "ACER", "AUC", "Accuracy", "Inference_ms"]:
        if key not in metrics:
            continue
        val  = metrics[key]
        name = display_names.get(key, key)

        if key == "Inference_ms":
            parts.append(f"{name}={val:.2f}ms")
        elif key in pct_keys:
            parts.append(f"{name}={val*100:.2f}%")
        else:
            parts.append(f"{name}={val:.4f}")

    return "  ".join(parts)


# =============================================================================
# SELF-TEST
# Run directly: python utils/metrics.py
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 65)
    print("METRICS SELF-TEST")
    print("=" * 65)

    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Generate realistic FAS score distributions
    # Real (label=0): scores clustered low
    # Spoof (label=1): scores clustered high
    # -------------------------------------------------------------------------
    n_samples = 500
    real_scores  = np.random.beta(2, 8, n_samples)     # concentrated near 0
    spoof_scores = np.random.beta(8, 2, n_samples)     # concentrated near 1

    y_true   = np.array([0] * n_samples + [1] * n_samples)
    y_scores = np.concatenate([real_scores, spoof_scores])

    print(f"\nTest data: {n_samples} real + {n_samples} spoof samples")
    print(f"Score range: [{y_scores.min():.4f}, {y_scores.max():.4f}]")

    # Test 1: EER
    print("\nTEST 1: EER")
    eer, thr = compute_eer(y_true, y_scores)
    print(f"  EER       : {eer*100:.4f}%")
    print(f"  Threshold : {thr:.4f}")
    assert 0.0 <= eer <= 1.0, "EER out of range"
    print("  PASS")

    # Test 2: HTER
    print("\nTEST 2: HTER")
    hter = compute_hter(y_true, y_scores, thr)
    print(f"  HTER      : {hter*100:.4f}%")
    assert 0.0 <= hter <= 1.0, "HTER out of range"
    print("  PASS")

    # Test 3: APCER / BPCER / ACER
    print("\nTEST 3: APCER / BPCER / ACER")
    apcer, bpcer, acer = compute_apcer_bpcer_acer(y_true, y_scores, thr)
    print(f"  APCER     : {apcer*100:.4f}%")
    print(f"  BPCER     : {bpcer*100:.4f}%")
    print(f"  ACER      : {acer*100:.4f}%")
    assert abs(acer - (apcer + bpcer) / 2) < 1e-6, "ACER formula wrong"
    print("  PASS")

    # Test 4: AUC
    print("\nTEST 4: AUC")
    auc_val = compute_auc(y_true, y_scores)
    print(f"  AUC       : {auc_val:.4f}")
    assert 0.5 <= auc_val <= 1.0, f"AUC {auc_val} unexpectedly low for separable data"
    print("  PASS")

    # Test 5: Accuracy
    print("\nTEST 5: Accuracy")
    acc = compute_accuracy(y_true, y_scores, thr)
    print(f"  Accuracy  : {acc*100:.4f}%")
    assert 0.0 <= acc <= 1.0, "Accuracy out of range"
    print("  PASS")

    # Test 6: compute_all_metrics
    print("\nTEST 6: compute_all_metrics (no model)")
    results = compute_all_metrics(y_true, y_scores)
    print(f"  Keys      : {list(results.keys())}")
    expected_keys = {"EER","HTER","APCER","BPCER","ACER","AUC","Accuracy","Inference_ms","_threshold"}
    assert set(results.keys()) == expected_keys, f"Missing keys: {expected_keys - set(results.keys())}"
    assert results["Inference_ms"] == -1.0, "Inference_ms should be -1.0 when no model"
    print("  PASS")

    # Test 7: compute_val_metric
    print("\nTEST 7: compute_val_metric")
    val_eer  = compute_val_metric(y_true, y_scores, "EER")
    val_hter = compute_val_metric(y_true, y_scores, "HTER")
    print(f"  Val EER   : {val_eer*100:.4f}%")
    print(f"  Val HTER  : {val_hter*100:.4f}%")
    assert abs(val_eer - results["EER"]) < 1e-6, "Val EER mismatch"
    print("  PASS")

    # Test 8: format_metrics_for_log
    print("\nTEST 8: format_metrics_for_log")
    formatted = format_metrics_for_log(results)
    print(f"  Output    : {formatted}")
    assert "EER=" in formatted, "EER missing from formatted output"
    assert "%" in formatted, "Percentage sign missing"
    print("  PASS")

    # Test 9: Perfect classifier sanity check
    print("\nTEST 9: Perfect classifier (EER should be ~0)")
    y_perfect = np.array([0]*100 + [1]*100)
    s_perfect = np.array([0.01]*100 + [0.99]*100)
    eer_p, _  = compute_eer(y_perfect, s_perfect)
    print(f"  EER (perfect): {eer_p*100:.4f}%")
    assert eer_p < 0.02, f"Perfect classifier EER too high: {eer_p}"
    print("  PASS")

    # Test 10: Random classifier sanity check
    print("\nTEST 10: Random classifier (EER should be ~50%)")
    np.random.seed(0)
    y_rand = np.array([0]*200 + [1]*200)
    s_rand = np.random.uniform(0, 1, 400)
    eer_r, _ = compute_eer(y_rand, s_rand)
    print(f"  EER (random): {eer_r*100:.4f}%")
    assert 0.35 <= eer_r <= 0.65, f"Random classifier EER unexpected: {eer_r}"
    print("  PASS")

    # Final summary
    print("\n" + "=" * 65)
    print("FULL METRICS SUMMARY ON TEST DATA:")
    print(format_metrics_for_log(results))
    print("=" * 65)
    print("\nALL 10 TESTS PASSED")
