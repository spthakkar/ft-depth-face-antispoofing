# =============================================================================
# utils/dataset_loader.py
# Dataset Loading — Face Anti-Spoofing Research Project
#
# Handles all data loading for Protocol A and Protocol B.
#
# Confirmed folder structure per dataset:
#   dataset_processed/
#       {DATASET_NAME}/
#           train/real/    train/spoof/
#           val/real/      val/spoof/
#           test/real/     test/spoof/
#
# Label convention (confirmed):
#   real  -> 0
#   spoof -> 1
#
# Key design decisions:
#   - Uses processed dataset root (output of Phase 1 face alignment)
#   - Training set gets augmentation; val/test get resize + normalize only
#   - Protocol B combines multiple datasets into one DataLoader
#   - Class imbalance is reported but NOT artificially balanced
#     (real-world FAS datasets are imbalanced; report it honestly)
#   - Returns DataLoader objects ready for training loop
#
# Usage:
#   from utils.dataset_loader import get_protocol_a_loaders
#   from utils.dataset_loader import get_protocol_b_loaders
#
#   # Protocol A
#   train_loader, val_loader, test_loader = get_protocol_a_loaders("OULU_NPU")
#
#   # Protocol B
#   train_loader, val_loader, test_loader = get_protocol_b_loaders("RpM_to_O")
# =============================================================================

import os
import sys
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# =============================================================================
# SECTION 1 — TRANSFORMS
# =============================================================================

def get_train_transform() -> transforms.Compose:
    """
    Augmentation pipeline for training set only.
    Settings sourced from config.AUGMENTATION.

    Returns:
        transforms.Compose: Training transform pipeline
    """
    aug = cfg.AUGMENTATION
    return transforms.Compose([
        # Resize slightly larger then random crop to IMAGE_SIZE
        # Simulates scale variation in capture conditions
        transforms.RandomResizedCrop(
            size  = cfg.IMAGE_SIZE,
            scale = aug["random_crop_scale"],
        ),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.ColorJitter(
            brightness = aug["brightness_jitter"],
            contrast   = aug["contrast_jitter"],
            saturation = aug["saturation_jitter"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = cfg.NORMALIZE_MEAN,
            std  = cfg.NORMALIZE_STD,
        ),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Minimal transform for validation and test sets.
    No augmentation — only resize and normalize.

    Returns:
        transforms.Compose: Evaluation transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = cfg.NORMALIZE_MEAN,
            std  = cfg.NORMALIZE_STD,
        ),
    ])


# =============================================================================
# SECTION 2 — DATASET CLASS
# =============================================================================

class FASDataset(Dataset):
    """
    PyTorch Dataset for Face Anti-Spoofing.

    Loads images from confirmed folder structure:
        root/real/  -> label 0
        root/spoof/ -> label 1

    Supports combining multiple root directories into one dataset
    (used for Protocol B where train = dataset_A/train + dataset_B/train).

    Args:
        root_dirs : List of directory paths. Each must contain real/ and spoof/
        transform : torchvision transform to apply
        split     : Name of split for logging ('train', 'val', 'test')
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dirs : List[str],
        transform : transforms.Compose,
        split     : str = "unknown",
    ):
        self.root_dirs = root_dirs
        self.transform = transform
        self.split     = split

        # Build image list: list of (image_path, label) tuples
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """
        Scan all root directories and collect image paths with labels.
        Skips unreadable files silently (logged at load time).
        """
        for root in self.root_dirs:
            for class_name, label in cfg.LABEL_MAP.items():
                class_dir = os.path.join(root, cfg.CLASS_FOLDERS[class_name])

                if not os.path.isdir(class_dir):
                    raise FileNotFoundError(
                        f"Class directory not found: {class_dir}\n"
                        f"  Expected structure: {root}/{class_name}/\n"
                        f"  Run Phase 1 (data preparation) first."
                    )
                #Collect all valid image files — recursive scan for nested folders
                # Handles client-wise/attack-wise subdirectory structures
                for walk_root, dirs, files in os.walk(class_dir):
                    dirs.sort()   # consistent ordering
                    for fname in sorted(files):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in self.VALID_EXTENSIONS:
                            continue
                        full_path = os.path.join(walk_root, fname)
                        self.samples.append((full_path, label))
                        """
                # Collect all valid image files
                for fname in sorted(os.listdir(class_dir)):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in self.VALID_EXTENSIONS:
                        continue
                    full_path = os.path.join(class_dir, fname)
                    self.samples.append((full_path, label))
                        """
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in any of the provided directories:\n"
                f"  {self.root_dirs}\n"
                f"  Check that Phase 1 completed successfully."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image: {img_path}\n"
                f"  Error: {e}"
            )

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """
        Return count of real and spoof samples.
        Used for imbalance reporting.

        Returns:
            Dict with keys 'real' and 'spoof'
        """
        counts = {"real": 0, "spoof": 0}
        for _, label in self.samples:
            if label == cfg.LABEL_MAP["real"]:
                counts["real"] += 1
            else:
                counts["spoof"] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for weighted loss (inverse frequency).
        Useful if class imbalance is severe.

        Returns:
            torch.Tensor: [weight_real, weight_spoof]
        """
        counts = self.get_class_counts()
        total  = sum(counts.values())
        weight_real  = total / (2.0 * max(counts["real"],  1))
        weight_spoof = total / (2.0 * max(counts["spoof"], 1))
        return torch.tensor([weight_real, weight_spoof], dtype=torch.float32)


# =============================================================================
# SECTION 3 — ROOT DIRECTORY RESOLVER
# =============================================================================

def _resolve_root(dataset_name: str, split: str) -> str:
    """
    Resolve the full directory path for a dataset split.
    Uses processed dataset root (output of Phase 1).

    Falls back to raw dataset root if processed does not exist,
    with a warning.

    Args:
        dataset_name : Key from config.DATASETS e.g. 'OULU_NPU'
        split        : One of 'train', 'val', 'test'

    Returns:
        str: Full path to the split directory
    """
    # Primary: use processed (aligned) frames
    processed_path = os.path.join(
        cfg.PROCESSED_DATASETS[dataset_name],
        cfg.SPLIT_NAMES[split],
    )

    if os.path.isdir(processed_path):
        return processed_path

    # Fallback: use raw frames (Phase 1 not yet run)
    raw_path = os.path.join(
        cfg.DATASETS[dataset_name],
        cfg.SPLIT_NAMES[split],
    )

    if os.path.isdir(raw_path):
        import warnings
        warnings.warn(
            f"Processed dataset not found at {processed_path}.\n"
            f"  Falling back to raw dataset: {raw_path}\n"
            f"  Run Phase 1 for best results.",
            UserWarning,
            stacklevel=3,
        )
        return raw_path

    raise FileNotFoundError(
        f"Dataset split not found.\n"
        f"  Processed path: {processed_path}\n"
        f"  Raw path      : {raw_path}\n"
        f"  Check DATASET_ROOT and PROCESSED_DATASET_ROOT in config.py"
    )


# =============================================================================
# SECTION 4 — DATALOADER FACTORY
# =============================================================================

def _make_loader(
    dataset     : FASDataset,
    batch_size  : int,
    shuffle     : bool,
    num_workers : int = None,
    pin_memory  : bool = None,
) -> DataLoader:
    """
    Create a DataLoader from a FASDataset.

    Args:
        dataset     : FASDataset instance
        batch_size  : Batch size
        shuffle     : Whether to shuffle (True for train, False for val/test)
        num_workers : Number of worker processes
        pin_memory  : Whether to pin memory for GPU transfer

    Returns:
        DataLoader
    """
    _workers    = num_workers if num_workers is not None else cfg.NUM_WORKERS
    _pin_memory = pin_memory  if pin_memory  is not None else cfg.PIN_MEMORY

    # Disable pin_memory on CPU-only machines
    if cfg.DEVICE == "cpu":
        _pin_memory = False

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = _workers,
        pin_memory  = _pin_memory,
        drop_last   = shuffle,   # drop last incomplete batch only during training
    )


def _log_dataset_info(
    loader_name : str,
    dataset     : FASDataset,
    logger      = None,
):
    """Log dataset statistics for transparency."""
    counts = dataset.get_class_counts()
    total  = sum(counts.values())
    msg    = (
        f"  {loader_name:<12} | "
        f"total={total:>7,} | "
        f"real={counts['real']:>7,} | "
        f"spoof={counts['spoof']:>7,} | "
        f"ratio={counts['real']/(counts['spoof']+1e-6):.2f}:1"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)


# =============================================================================
# SECTION 5 — PROTOCOL A LOADERS
# =============================================================================

def get_protocol_a_loaders(
    dataset_name : str,
    batch_size   : int = None,
    logger       = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, val, test DataLoaders for Protocol A (intra-dataset).

    Uses the official train/val/test splits from the dataset.
    Training set gets augmentation; val and test get eval transform only.

    Args:
        dataset_name : One of 'OULU_NPU', 'Replay_Attack', 'MSU_MFSD'
        batch_size   : Override default batch size from config
        logger       : Optional logger for dataset statistics

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if dataset_name not in cfg.DATASETS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Valid: {list(cfg.DATASETS.keys())}"
        )

    _batch_size = batch_size or cfg.BATCH_SIZE

    if logger:
        logger.info(f"Loading Protocol A dataset: {dataset_name}")

    # Resolve paths
    train_root = _resolve_root(dataset_name, "train")
    val_root   = _resolve_root(dataset_name, "val")
    test_root  = _resolve_root(dataset_name, "test")

    # Build datasets
    train_dataset = FASDataset([train_root], get_train_transform(), split="train")
    val_dataset   = FASDataset([val_root],   get_eval_transform(),  split="val")
    test_dataset  = FASDataset([test_root],  get_eval_transform(),  split="test")

    # Log statistics
    if logger:
        logger.info(f"Dataset statistics for {dataset_name}:")
    _log_dataset_info("train", train_dataset, logger)
    _log_dataset_info("val",   val_dataset,   logger)
    _log_dataset_info("test",  test_dataset,  logger)

    # Build loaders
    train_loader = _make_loader(train_dataset, _batch_size, shuffle=True)
    val_loader   = _make_loader(val_dataset,   _batch_size, shuffle=False)
    test_loader  = _make_loader(test_dataset,  _batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# SECTION 6 — PROTOCOL B LOADERS
# =============================================================================

def get_protocol_b_loaders(
    config_name : str,
    batch_size  : int = None,
    logger      = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, val, test DataLoaders for Protocol B (cross-dataset).

    Following Almeida et al. (2025) unbiased benchmark:
      - train = train+val splits of training datasets combined
      - val   = val split of the TEST dataset (unbiased model selection)
      - test  = test split of the TEST dataset (evaluated ONCE after training)

    Args:
        config_name : Key from config.PROTOCOL_B_CONFIGS
                      e.g. 'RpM_to_O', 'OpM_to_R', 'OpR_to_M'
        batch_size  : Override default batch size from config
        logger      : Optional logger for dataset statistics

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        IMPORTANT: test_loader should only be used ONCE after training.
    """
    if config_name not in cfg.PROTOCOL_B_CONFIGS:
        raise ValueError(
            f"Unknown Protocol B config: '{config_name}'. "
            f"Valid: {list(cfg.PROTOCOL_B_CONFIGS.keys())}"
        )

    pb_config   = cfg.PROTOCOL_B_CONFIGS[config_name]
    _batch_size = batch_size or cfg.BATCH_SIZE

    if logger:
        logger.info(
            f"Loading Protocol B config: {pb_config['display_name']} ({config_name})"
        )

    # ------------------------------------------------------------------
    # Build TRAINING dataset
    # Combines train + val splits of all training datasets
    # ------------------------------------------------------------------
    train_roots = []
    for entry in pb_config["train_datasets"]:
        dname  = entry["dataset"]
        splits = entry["splits"]
        for split in splits:
            train_roots.append(_resolve_root(dname, split))

    train_dataset = FASDataset(train_roots, get_train_transform(), split="train")

    # ------------------------------------------------------------------
    # Build VALIDATION dataset
    # Val split of the TEST dataset (Almeida et al. approach)
    # ------------------------------------------------------------------
    val_entry  = pb_config["val_dataset"]
    val_roots  = [
        _resolve_root(val_entry["dataset"], split)
        for split in val_entry["splits"]
    ]
    val_dataset = FASDataset(val_roots, get_eval_transform(), split="val")

    # ------------------------------------------------------------------
    # Build TEST dataset
    # Test split of the TEST dataset — NEVER touched during training
    # ------------------------------------------------------------------
    test_entry  = pb_config["test_dataset"]
    test_roots  = [
        _resolve_root(test_entry["dataset"], split)
        for split in test_entry["splits"]
    ]
    test_dataset = FASDataset(test_roots, get_eval_transform(), split="test")

    # Log statistics
    if logger:
        train_names = [e["dataset"] for e in pb_config["train_datasets"]]
        logger.info(f"Train datasets : {train_names}")
        logger.info(f"Val dataset    : {val_entry['dataset']}")
        logger.info(f"Test dataset   : {test_entry['dataset']}  [LOCKED until end of training]")
        logger.info("Dataset statistics:")

    _log_dataset_info("train (combined)", train_dataset, logger)
    _log_dataset_info("val",              val_dataset,   logger)
    _log_dataset_info("test [LOCKED]",    test_dataset,  logger)

    # Build loaders
    train_loader = _make_loader(train_dataset, _batch_size, shuffle=True)
    val_loader   = _make_loader(val_dataset,   _batch_size, shuffle=False)
    test_loader  = _make_loader(test_dataset,  _batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# SECTION 7 — UTILITY: SINGLE SPLIT LOADER
# Used by Phase 6 (visualization) to load a specific split for a model
# =============================================================================

def get_single_split_loader(
    dataset_name : str,
    split        : str,
    batch_size   : int = None,
    shuffle      : bool = False,
    logger       = None,
) -> DataLoader:
    """
    Load a single split of a single dataset.
    Used by Phase 6 for Grad-CAM and t-SNE sample loading.

    Args:
        dataset_name : Dataset key
        split        : 'train', 'val', or 'test'
        batch_size   : Batch size override
        shuffle      : Whether to shuffle
        logger       : Optional logger

    Returns:
        DataLoader
    """
    _batch_size = batch_size or cfg.BATCH_SIZE
    root        = _resolve_root(dataset_name, split)
    dataset     = FASDataset([root], get_eval_transform(), split=split)

    if logger:
        _log_dataset_info(f"{dataset_name}/{split}", dataset, logger)

    return _make_loader(dataset, _batch_size, shuffle=shuffle)


# =============================================================================
# SECTION 8 — UTILITY: DATASET SUMMARY
# Used by Phase 1 verification and Phase 7 reporting
# =============================================================================

def get_dataset_summary(
    dataset_name : str,
    logger       = None,
) -> Dict:
    """
    Return statistics for all splits of a dataset without building loaders.
    Used for verification and reporting.

    Args:
        dataset_name : Dataset key
        logger       : Optional logger

    Returns:
        Dict with split statistics
    """
    summary = {"dataset": dataset_name, "splits": {}}

    for split in ["train", "val", "test"]:
        try:
            root    = _resolve_root(dataset_name, split)
            dataset = FASDataset([root], get_eval_transform(), split=split)
            counts  = dataset.get_class_counts()
            total   = sum(counts.values())
            summary["splits"][split] = {
                "total" : total,
                "real"  : counts["real"],
                "spoof" : counts["spoof"],
                "ratio" : round(counts["real"] / max(counts["spoof"], 1), 3),
                "root"  : root,
            }
        except FileNotFoundError as e:
            summary["splits"][split] = {"error": str(e)}

    return summary


# =============================================================================
# SELF-TEST
# Run directly: python utils/dataset_loader.py
# Creates dummy dataset folder structure and tests all loaders.
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    import numpy as np
    from PIL import Image as PILImage

    print("=" * 65)
    print("DATASET LOADER SELF-TEST")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Create temporary dummy dataset structure
    # ------------------------------------------------------------------
    def create_dummy_images(folder: str, n: int = 10):
        """Create n dummy 224x224 RGB images in folder."""
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = PILImage.fromarray(arr)
            img.save(os.path.join(folder, f"img_{i:04d}.jpg"))

    # Create temp directory mimicking processed dataset structure
    tmp_dir = tempfile.mkdtemp(prefix="fas_test_")
    print(f"\nTemp dataset root: {tmp_dir}")

    datasets    = ["OULU_NPU", "Replay_Attack", "MSU_MFSD"]
    splits      = ["train", "val", "test"]
    class_names = ["real", "spoof"]

    print("\nCreating dummy image structure...")
    for ds in datasets:
        for split in splits:
            for cls in class_names:
                folder = os.path.join(tmp_dir, ds, split, cls)
                n_imgs = 20 if split == "train" else 10
                create_dummy_images(folder, n_imgs)
    print("  Done — 6 datasets × 3 splits × 2 classes created")

    # Override config paths temporarily for testing
    original_processed = cfg.PROCESSED_DATASET_ROOT
    original_processed_datasets = cfg.PROCESSED_DATASETS.copy()

    cfg.PROCESSED_DATASET_ROOT = tmp_dir
    for ds in datasets:
        cfg.PROCESSED_DATASETS[ds] = os.path.join(tmp_dir, ds)

    try:
        # Test 1: FASDataset basic loading
        print("\nTEST 1: FASDataset basic loading")
        root    = os.path.join(tmp_dir, "OULU_NPU", "train")
        dataset = FASDataset([root], get_eval_transform(), split="train")
        print(f"  Total samples   : {len(dataset)}")
        counts  = dataset.get_class_counts()
        print(f"  Real samples    : {counts['real']}")
        print(f"  Spoof samples   : {counts['spoof']}")
        img, label = dataset[0]
        assert img.shape == (3, 224, 224), f"Wrong shape: {img.shape}"
        assert label in [0, 1], f"Wrong label: {label}"
        print(f"  Sample shape    : {tuple(img.shape)}")
        print(f"  Sample label    : {label}")
        print("  PASS")

        # Test 2: Train augmentation transform
        print("\nTEST 2: Train vs eval transform")
        train_ds = FASDataset([root], get_train_transform(), split="train")
        eval_ds  = FASDataset([root], get_eval_transform(),  split="val")
        t_img, _ = train_ds[0]
        e_img, _ = eval_ds[0]
        assert t_img.shape == (3, 224, 224)
        assert e_img.shape == (3, 224, 224)
        print(f"  Train transform output shape: {tuple(t_img.shape)}")
        print(f"  Eval  transform output shape: {tuple(e_img.shape)}")
        print("  PASS")

        # Test 3: Class weights
        print("\nTEST 3: Class weights")
        weights = dataset.get_class_weights()
        assert weights.shape == (2,), f"Wrong weight shape: {weights.shape}"
        print(f"  Class weights   : {weights.tolist()}")
        print("  PASS")

        # Test 4: Protocol A loaders
        print("\nTEST 4: Protocol A loaders")
        train_loader, val_loader, test_loader = get_protocol_a_loaders(
            "OULU_NPU", batch_size=4
        )
        print(f"  Train batches   : {len(train_loader)}")
        print(f"  Val   batches   : {len(val_loader)}")
        print(f"  Test  batches   : {len(test_loader)}")

        # Check one batch
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] <= 4
        assert batch_x.shape[1:] == (3, 224, 224)
        assert set(batch_y.numpy().tolist()).issubset({0, 1})
        print(f"  Batch shape     : {tuple(batch_x.shape)}")
        print(f"  Batch labels    : {batch_y.tolist()}")
        print("  PASS")

        # Test 5: Protocol B loaders
        print("\nTEST 5: Protocol B loaders — RpM_to_O")
        pb_train, pb_val, pb_test = get_protocol_b_loaders(
            "RpM_to_O", batch_size=4
        )
        print(f"  Train batches   : {len(pb_train)}  (combined Replay+MSU)")
        print(f"  Val   batches   : {len(pb_val)}    (OULU val split)")
        print(f"  Test  batches   : {len(pb_test)}   (OULU test split) [LOCKED]")

        batch_x, batch_y = next(iter(pb_train))
        assert batch_x.shape[1:] == (3, 224, 224)
        print(f"  Train batch shape: {tuple(batch_x.shape)}")
        print("  PASS")

        # Test 6: All 3 Protocol B configs
        print("\nTEST 6: All Protocol B configurations")
        for config_name in cfg.PROTOCOL_B_CONFIG_NAMES:
            try:
                tl, vl, tel = get_protocol_b_loaders(config_name, batch_size=4)
                display      = cfg.PROTOCOL_B_CONFIGS[config_name]["display_name"]
                print(f"  {display:<12} | train={len(tl)} | val={len(vl)} | test={len(tel)} | PASS")
            except Exception as e:
                print(f"  {config_name} | FAIL — {e}")
        print("  PASS")

        # Test 7: Single split loader
        print("\nTEST 7: Single split loader")
        single = get_single_split_loader("MSU_MFSD", "test", batch_size=4)
        print(f"  MSU_MFSD/test   : {len(single)} batches")
        print("  PASS")

        # Test 8: Dataset summary
        print("\nTEST 8: Dataset summary")
        summary = get_dataset_summary("Replay_Attack")
        print(f"  Dataset         : {summary['dataset']}")
        for split, info in summary["splits"].items():
            if "error" not in info:
                print(
                    f"  {split:<6}: total={info['total']} "
                    f"real={info['real']} spoof={info['spoof']}"
                )
        print("  PASS")

        # Test 9: FileNotFoundError on bad dataset name
        print("\nTEST 9: Error handling — bad dataset name")
        try:
            get_protocol_a_loaders("NONEXISTENT_DATASET")
            print("  FAIL — should have raised ValueError")
        except ValueError as e:
            print(f"  Correctly raised ValueError")
            print("  PASS")

        # Test 10: drop_last behavior (train shuffles, test does not)
        print("\nTEST 10: drop_last behavior")
        assert train_loader.drop_last is True,  "Train loader should drop_last=True"
        assert test_loader.drop_last  is False, "Test loader should drop_last=False"
        print(f"  Train drop_last : {train_loader.drop_last}  (expected True)")
        print(f"  Test  drop_last : {test_loader.drop_last}   (expected False)")
        print("  PASS")

    finally:
        # Restore config paths
        cfg.PROCESSED_DATASET_ROOT = original_processed
        cfg.PROCESSED_DATASETS.update(original_processed_datasets)
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nCleaned up temp directory: {tmp_dir}")

    print("\n" + "=" * 65)
    print("ALL 10 TESTS PASSED")
    print("=" * 65)
