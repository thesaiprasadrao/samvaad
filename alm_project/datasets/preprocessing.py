"""
Audio preprocessing pipeline for ALM project.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split

from ..utils.audio_utils import AudioUtils
from ..utils.config import Config


class AudioPreprocessor:
    """Audio preprocessing pipeline for ALM datasets."""
    
    def __init__(self, config: Config):
        """Initialize preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audio_utils = AudioUtils()
        
        # Get configuration values
        self.sample_rate = config.get('data.sample_rate', 16000)
        self.max_duration = config.get('data.max_duration', 10.0)
        self.train_split = config.get('data.train_split', 0.7)
        self.val_split = config.get('data.val_split', 0.15)
        self.test_split = config.get('data.test_split', 0.15)
    
    def load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load metadata from CSV file.
        
        Args:
            metadata_file: Path to metadata CSV file
            
        Returns:
            DataFrame with metadata
        """
        try:
            df = pd.read_csv(metadata_file)
            self.logger.info(f"Loaded {len(df)} records from {metadata_file}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            raise
    
    def preprocess_audio_file(
        self,
        file_path: str,
        target_duration: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """Preprocess single audio file.
        
        Args:
            file_path: Path to audio file
            target_duration: Target duration in seconds
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        try:
            # Load audio
            audio, sr = self.audio_utils.load_audio(file_path, self.sample_rate)
            
            # Normalize audio
            audio = self.audio_utils.normalize_audio(audio)
            
            # Trim silence
            audio = self.audio_utils.trim_silence(audio, sr)
            
            # Pad or truncate to target duration
            if target_duration is None:
                target_duration = self.max_duration
            
            target_length = int(target_duration * self.sample_rate)
            audio = self.audio_utils.pad_or_truncate(audio, target_length)
            
            # Get audio info
            audio_info = {
                'duration': len(audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'length': len(audio)
            }
            
            return audio, audio_info
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {file_path}: {e}")
            raise
    
    def create_splits(
        self,
        df: pd.DataFrame,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits.
        
        Args:
            df: Input DataFrame
            stratify_column: Column to stratify on for balanced splits
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train vs (val + test)
        train_size = self.train_split
        val_test_size = self.val_split + self.test_split
        
        if stratify_column and stratify_column in df.columns:
            train_df, val_test_df = train_test_split(
                df,
                train_size=train_size,
                test_size=val_test_size,
                stratify=df[stratify_column],
                random_state=42
            )
        else:
            train_df, val_test_df = train_test_split(
                df,
                train_size=train_size,
                test_size=val_test_size,
                random_state=42
            )
        
        # Second split: val vs test
        val_size = self.val_split / (self.val_split + self.test_split)
        
        if stratify_column and stratify_column in df.columns:
            val_df, test_df = train_test_split(
                val_test_df,
                train_size=val_size,
                test_size=1-val_size,
                stratify=val_test_df[stratify_column],
                random_state=42
            )
        else:
            val_df, test_df = train_test_split(
                val_test_df,
                train_size=val_size,
                test_size=1-val_size,
                random_state=42
            )
        
        self.logger.info(f"Created splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def filter_by_duration(
        self,
        df: pd.DataFrame,
        min_duration: float = 0.5,
        max_duration: Optional[float] = None
    ) -> pd.DataFrame:
        """Filter dataset by audio duration.
        
        Args:
            df: Input DataFrame
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            Filtered DataFrame
        """
        if max_duration is None:
            max_duration = self.max_duration
        
        # Add duration column if not present
        if 'duration' not in df.columns:
            durations = []
            for _, row in df.iterrows():
                try:
                    file_path = Path(row['filepath'])
                    if file_path.exists():
                        info = self.audio_utils.get_audio_info(file_path)
                        durations.append(info['duration'])
                    else:
                        durations.append(0)
                except:
                    durations.append(0)
            df['duration'] = durations
        
        # Ensure duration column is numeric
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        
        # Filter by duration
        mask = (df['duration'] >= min_duration) & (df['duration'] <= max_duration)
        filtered_df = df[mask].copy()
        
        self.logger.info(f"Filtered {len(df)} -> {len(filtered_df)} files by duration")
        
        return filtered_df
    
    def create_balanced_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_samples_per_class: Optional[int] = None
    ) -> pd.DataFrame:
        """Create balanced dataset by sampling from each class.
        
        Args:
            df: Input DataFrame
            target_column: Column to balance on
            max_samples_per_class: Maximum samples per class
            
        Returns:
            Balanced DataFrame
        """
        balanced_dfs = []
        
        for class_name in df[target_column].unique():
            class_df = df[df[target_column] == class_name]
            
            if max_samples_per_class and len(class_df) > max_samples_per_class:
                class_df = class_df.sample(n=max_samples_per_class, random_state=42)
            
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.logger.info(f"Created balanced dataset: {len(balanced_df)} samples")
        self.logger.info(f"Class distribution:\n{balanced_df[target_column].value_counts()}")
        
        return balanced_df
    
    def preprocess_dataset(
        self,
        metadata_file: str,
        output_dir: str = "processed_data",
        preprocess_audio: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Preprocess entire dataset.
        
        Args:
            metadata_file: Path to metadata CSV file
            output_dir: Output directory for processed data
            preprocess_audio: Whether to preprocess audio files
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        df = self.load_metadata(metadata_file)
        
        # Filter by duration
        df = self.filter_by_duration(df)
        
        # Create splits
        train_df, val_df, test_df = self.create_splits(df, stratify_column='type')
        
        # Save splits
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        self.logger.info(f"Saved dataset splits to {output_dir}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def get_class_weights(self, df: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Calculate class weights for imbalanced datasets.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            
        Returns:
            Dictionary of class weights
        """
        class_counts = df[target_column].value_counts()
        total_samples = len(df)
        
        weights = {}
        for class_name, count in class_counts.items():
            weights[class_name] = total_samples / (len(class_counts) * count)
        
        return weights
