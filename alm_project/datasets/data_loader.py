"""
Data loading utilities for ALM project.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader, WeightedRandomSampler
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from .audio_dataset import AudioDataset
from .preprocessing import AudioPreprocessor
from ..utils.config import Config


class DataLoader:
    """Data loading utilities for ALM project."""
    
    def __init__(self, config: Config):
        """Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.preprocessor = AudioPreprocessor(config)
    
    def create_dataloader(
        self,
        metadata_df: pd.DataFrame,
        root_dir: str,
        task: str,
        split: str = "train",
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        use_weighted_sampling: bool = False
    ) -> TorchDataLoader:
        """Create PyTorch DataLoader.
        
        Args:
            metadata_df: DataFrame with metadata
            root_dir: Root directory for audio files
            task: Task type ('transcription', 'emotion', 'cultural_context')
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle (uses split-based default if None)
            use_weighted_sampling: Whether to use weighted sampling for imbalanced data
            
        Returns:
            PyTorch DataLoader
        """
        # Get configuration values
        if batch_size is None:
            batch_size = self.config.get(f'models.{task}.batch_size', 16)
        
        if shuffle is None:
            shuffle = (split == "train")
        
        # Create dataset
        dataset = AudioDataset(
            metadata_df=metadata_df,
            root_dir=root_dir,
            config=self.config.config,
            task=task,
            preprocessor=self.preprocessor
        )
        
        # Create sampler for weighted sampling
        sampler = None
        if use_weighted_sampling and split == "train" and task != "transcription":
            class_weights = dataset.get_class_weights()
            if class_weights is not None and len(class_weights) > 0:
                # Get class indices for each sample
                class_indices = []
                for _, row in metadata_df.iterrows():
                    label_text = row.get(dataset.label_column, "")
                    class_id = dataset.label_map.get(label_text, -1)
                    if class_id >= 0 and class_id < len(class_weights):
                        class_indices.append(class_id)
                
                # Create sample weights
                if class_indices:
                    sample_weights = [class_weights[i] for i in class_indices]
                    sampler = WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(dataset),
                        replacement=True
                    )
                    shuffle = False  # Don't shuffle when using sampler
        
        # Create DataLoader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=(split == "train")
        )
        
        self.logger.info(f"Created {task} {split} DataLoader with {len(dataset)} samples")
        
        return dataloader
    
    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        root_dir: str,
        task: str,
        batch_size: Optional[int] = None
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """Create train/val/test DataLoaders.
        
        Args:
            train_df: Training metadata DataFrame
            val_df: Validation metadata DataFrame
            test_df: Test metadata DataFrame
            root_dir: Root directory for audio files
            task: Task type
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create DataLoaders
        train_loader = self.create_dataloader(
            train_df, root_dir, task, "train", batch_size,
            use_weighted_sampling=True
        )
        
        val_loader = self.create_dataloader(
            val_df, root_dir, task, "val", batch_size
        )
        
        test_loader = self.create_dataloader(
            test_df, root_dir, task, "test", batch_size
        )
        
        return train_loader, val_loader, test_loader
    
    def load_processed_data(
        self,
        data_dir: str,
        task: str,
        batch_size: Optional[int] = None
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """Load processed dataset splits.
        
        Args:
            data_dir: Directory containing processed data
            task: Task type
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_path = Path(data_dir)
        
        # Load metadata files
        print(f"Loading data from {data_path}")
        train_df = pd.read_csv(data_path / "train.csv")
        val_df = pd.read_csv(data_path / "val.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        
        print(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
        
        # Get root directory from config
        root_dir = self.config.get('data.root_dir', '.')
        print(f"Using root directory: {root_dir}")
        
        # Create DataLoaders
        return self.create_dataloaders(
            train_df, val_df, test_df, root_dir, task, batch_size
        )
    
    def get_dataset_stats(self, metadata_df: pd.DataFrame, task: str) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Args:
            metadata_df: Metadata DataFrame
            task: Task type
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(metadata_df),
            'task': task
        }
        
        if task == "transcription":
            # Transcription stats
            stats['languages'] = metadata_df['language'].value_counts().to_dict()
            stats['datasets'] = metadata_df['dataset'].value_counts().to_dict()
        else:
            # Classification task stats
            if task == "emotion":
                label_col = "emotion"
            elif task == "cultural_context":
                label_col = "type"
            else:
                label_col = "label"
            
            if label_col in metadata_df.columns:
                stats['class_distribution'] = metadata_df[label_col].value_counts().to_dict()
                stats['num_classes'] = len(metadata_df[label_col].unique())
        
        return stats
