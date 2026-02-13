"""
PyTorch Dataset for Phase A training.

Loads consecutive frame pairs (x_t, x_{t+1}) from the TDW-generated dataset.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PoseSequenceDataset(Dataset):
    """
    Dataset that loads frame pairs (x_t, x_{t+delta}) for training.
    
    Expected directory structure (from create_dataset.py):
        data_root/
            dataset_index.json
            triples.json
            train/
                model_name/
                    frames/a/img_0000.png, img_0001.png, ...
                    meta.jsonl
            test/
                model_name/
                    frames/a/img_0000.png, ...
    
    From Training Protocol (10.4):
    - Use a single fixed step size (Angela suggests 6° or 4°, don't mix)
    - frame_skip controls angular step: skip=1 → consecutive, skip=3 → every 3rd frame
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        object_name: Optional[str] = None,
        img_size: int = 256,
        frame_skip: int = 1,
        center_crop: Optional[int] = None,
        augment: bool = False,
        include_backward: bool = False,
    ):
        """
        Args:
            data_root: Path to dataset root (contains dataset_index.json)
            split: "train" or "test"
            object_name: If specified, only load this object. Otherwise load all.
            img_size: Resize images to this size
            frame_skip: Number of frames to skip between pairs.
                        If yaw_step=2° in dataset: skip=1→2°, skip=2→4°, skip=3→6°
            center_crop: If specified, center crop to this size before resize
            augment: Whether to apply data augmentation (not recommended for Phase A)
            include_backward: If True, add reversed pairs for action classification
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.include_backward = include_backward
        
        # Load dataset index
        index_path = self.data_root / "dataset_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {index_path}")
        
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Determine which objects to load
        if object_name:
            self.objects = [object_name]
        else:
            self.objects = self.index["splits"].get(split, [])
        
        if not self.objects:
            raise ValueError(f"No objects found for split '{split}'")
        
        # Build list of all frame pairs with frame_skip
        self.pairs = []
        for obj in self.objects:
            obj_dir = self.data_root / split / obj / "frames" / "a"
            if not obj_dir.exists():
                print(f"Warning: Object directory not found: {obj_dir}")
                continue
            
            # Get sorted list of frame files
            frames = sorted(obj_dir.glob("img_*.png"))
            
            # Create pairs with frame_skip
            for i in range(len(frames) - frame_skip):
                self.pairs.append({
                    'x_t_path': frames[i],
                    'x_t1_path': frames[i + frame_skip],
                    'object': obj,
                    't': i,
                    't1': i + frame_skip,
                    'action_label': 0,  # forward (yaw+)
                })
                if include_backward:
                    self.pairs.append({
                        'x_t_path': frames[i + frame_skip],
                        'x_t1_path': frames[i],
                        'object': obj,
                        't': i + frame_skip,
                        't1': i,
                        'action_label': 1,  # backward (yaw-)
                    })
        
        if not self.pairs:
            raise ValueError(
                f"Dataset produced 0 pairs for split='{split}', objects={self.objects}, "
                f"frame_skip={frame_skip}. Check that frame directories exist and contain "
                f"enough frames for the requested frame_skip."
            )
        
        print(
            f"Loaded {len(self.pairs)} pairs from {len(self.objects)} objects "
            f"({split}, frame_skip={frame_skip}, include_backward={include_backward})"
        )
        
        # Image transforms
        transform_list = []
        if center_crop is not None:
            transform_list.append(transforms.CenterCrop(center_crop))
        transform_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        
        # Load images
        x_t = Image.open(pair['x_t_path']).convert('RGB')
        x_t1 = Image.open(pair['x_t1_path']).convert('RGB')
        
        # Apply transforms
        x_t = self.transform(x_t)
        x_t1 = self.transform(x_t1)
        
        return {
            'x_t': x_t,
            'x_t1': x_t1,
            'object': pair['object'],
            't': pair['t'],
            't1': pair['t1'],
            'action_label': pair['action_label'],
        }


class SingleObjectDataset(Dataset):
    """
    Simplified dataset for single-object training (recommended for Phase A debugging).
    
    Directly loads frames from one object's directory.
    
    From Training Protocol (10.4):
    - Use a single fixed step size (Angela suggests 6° or 4°, don't mix)
    - frame_skip controls the angular step: skip=1 → consecutive, skip=3 → every 3rd frame
    """
    
    def __init__(
        self,
        frames_dir: str,
        img_size: int = 256,
        frame_skip: int = 1,
        center_crop: Optional[int] = None,
        include_backward: bool = False,
    ):
        """
        Args:
            frames_dir: Path to frames directory (e.g., data/train/coffeemug/frames/a/)
            img_size: Resize images to this size
            frame_skip: Number of frames to skip between pairs.
                        If yaw_step=2° in dataset: skip=1→2°, skip=2→4°, skip=3→6°
            center_crop: If specified, center crop to this size before resize (reduces background)
            include_backward: If True, add reversed pairs for action classification
        """
        self.frames_dir = Path(frames_dir)
        self.frame_skip = frame_skip
        self.include_backward = include_backward
        
        # Get sorted list of frame files
        self.frames = sorted(self.frames_dir.glob("img_*.png"))
        if not self.frames:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        
        n_pairs = len(self.frames) - frame_skip
        if n_pairs <= 0:
            raise ValueError(
                f"Dataset produced 0 pairs: {len(self.frames)} frames with "
                f"frame_skip={frame_skip}. Need at least {frame_skip + 1} frames."
            )
        self.n_forward = n_pairs
        n_total = n_pairs * (2 if include_backward else 1)
        print(
            f"Loaded {len(self.frames)} frames, frame_skip={frame_skip} -> "
            f"{self.n_forward} forward pairs ({n_total} total, include_backward={include_backward})"
        )
        
        # Image transforms
        transform_list = []
        if center_crop is not None:
            transform_list.append(transforms.CenterCrop(center_crop))
        transform_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        if self.include_backward:
            return self.n_forward * 2
        return self.n_forward

    def __getitem__(self, idx: int) -> dict:
        if idx < self.n_forward:
            t = idx
            t1 = idx + self.frame_skip
            action_label = 0
        else:
            rev_idx = idx - self.n_forward
            t = rev_idx + self.frame_skip
            t1 = rev_idx
            action_label = 1

        # Load frames with skip (forward or backward)
        x_t = Image.open(self.frames[t]).convert('RGB')
        x_t1 = Image.open(self.frames[t1]).convert('RGB')
        
        # Apply transforms
        x_t = self.transform(x_t)
        x_t1 = self.transform(x_t1)
        
        return {
            'x_t': x_t,
            'x_t1': x_t1,
            't': t,
            't1': t1,
            'action_label': action_label,
        }


def get_inverse_transform():
    """Get transform to convert normalized tensor back to displayable image."""
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
    ])


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_root>")
        print("Example: python dataset.py ./phase_a_yaw_only")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    # Test full dataset
    try:
        dataset = PoseSequenceDataset(data_root, split="train")
        sample = dataset[0]
        print(f"Sample x_t shape: {sample['x_t'].shape}")
        print(f"Sample x_t1 shape: {sample['x_t1'].shape}")
        print(f"Sample object: {sample['object']}")
        print(f"Sample t: {sample['t']}")
        print(f"Sample action_label: {sample['action_label']}")
    except Exception as e:
        print(f"Could not load full dataset: {e}")
    
    # Test single object dataset (if path provided)
    if len(sys.argv) > 2:
        frames_dir = sys.argv[2]
        dataset = SingleObjectDataset(frames_dir)
        sample = dataset[0]
        print(f"Single object - x_t shape: {sample['x_t'].shape}")
        print(f"Single object - action_label: {sample['action_label']}")
