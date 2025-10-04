"""
PyTorch Dataset class for ALM audio data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from .preprocessing import AudioPreprocessor
from ..utils.audio_utils import AudioUtils


class AudioDataset(Dataset):
    """PyTorch Dataset for ALM audio data."""
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        root_dir: str,
        config: Dict[str, Any],
        task: str = "transcription",
        preprocessor: Optional[AudioPreprocessor] = None
    ):
        """Initialize dataset.
        
        Args:
            metadata_df: DataFrame with metadata
            root_dir: Root directory for audio files
            config: Configuration dictionary
            task: Task type ('transcription', 'emotion', 'cultural_context')
            preprocessor: Audio preprocessor instance
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.config = config
        self.task = task
        self.preprocessor = preprocessor or AudioPreprocessor(config)
        self.audio_utils = AudioUtils()
        
        # Task-specific configuration
        self.sample_rate = config.get('data.sample_rate', 16000)
        self.max_duration = config.get('data.max_duration', 10.0)
        
        # Task-specific labels
        if task == "emotion":
            self.label_column = "emotion"
            self.label_map = {
                "anger": 0, "disgust": 1, "fear": 2, 
                "happiness": 3, "sadness": 4
            }
            
            # Filter out samples with NaN emotion values
            self.metadata_df = self.metadata_df.dropna(subset=['emotion'])
            print(f"After filtering NaN emotions: {len(self.metadata_df)} samples")
            print(f"Emotion column unique values: {self.metadata_df['emotion'].unique()}")
            print(f"Emotion column value counts: {self.metadata_df['emotion'].value_counts()}")
        elif task == "cultural_context":
            self.label_column = "type"
            self.label_map = {
                "speech": 0, "non-speech": 1
            }
        else:  # transcription
            self.label_column = "transcription"
            self.label_map = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized {task} dataset with {len(self.metadata_df)} samples")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with audio tensor and labels
        """
        try:
            row = self.metadata_df.iloc[idx]
            
            # Load and preprocess audio
            audio_path = self.root_dir / row['filepath']
            
            try:
                # Load audio
                audio, sr = self.audio_utils.load_audio(
                    audio_path, 
                    sample_rate=self.sample_rate
                )
                
                # Normalize and trim
                audio = self.audio_utils.normalize_audio(audio)
                audio = self.audio_utils.trim_silence(audio, sr)
                
                # Pad or truncate
                target_length = int(self.max_duration * self.sample_rate)
                audio = self.audio_utils.pad_or_truncate(audio, target_length)
                
                # Convert to tensor
                audio_tensor = self.audio_utils.audio_to_tensor(audio)
                
            except Exception as e:
                self.logger.warning(f"Error loading audio {audio_path}: {e}")
                # Return zero tensor as fallback
                audio_tensor = torch.zeros(1, int(self.max_duration * self.sample_rate))
        except Exception as e:
            self.logger.error(f"Error processing sample {idx}: {e}")
            # Return dummy data
            audio_tensor = torch.zeros(1, int(self.max_duration * self.sample_rate))
            row = self.metadata_df.iloc[0]  # Use first row as fallback
        
        # Prepare labels based on task
        if self.task == "transcription":
            # For transcription, return text label
            label = row.get(self.label_column, "")
            return {
                'audio': audio_tensor,
                'transcription': label,
                'filepath': str(audio_path)
            }
        
        elif self.task in ["emotion", "cultural_context"]:
            # For classification tasks, return numeric label
            label_text = row.get(self.label_column, "")
            label_id = self.label_map.get(label_text, -1)
            
            # Skip samples with unknown labels
            if label_id == -1:
                # Return a dummy sample with label 0
                label_id = 0
                label_text = list(self.label_map.keys())[0]
            
            return {
                'audio': audio_tensor,
                'label': torch.tensor(label_id, dtype=torch.long),
                'label_text': label_text,
                'filepath': str(audio_path)
            }
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for imbalanced datasets.
        
        Returns:
            Tensor with class weights
        """
        if self.task == "transcription":
            return None
        
        # Count samples per class
        class_counts = self.metadata_df[self.label_column].value_counts()
        total_samples = len(self.metadata_df)
        num_classes = len(class_counts)
        
        # Calculate weights
        weights = torch.zeros(num_classes)
        for class_name, count in class_counts.items():
            if class_name in self.label_map:
                class_id = self.label_map[class_name]
                weights[class_id] = total_samples / (num_classes * count)
        
        return weights
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution.
        
        Returns:
            Dictionary with class counts
        """
        if self.task == "transcription":
            return {}
        
        return self.metadata_df[self.label_column].value_counts().to_dict()
    
    def filter_by_class(self, class_names: list) -> 'AudioDataset':
        """Filter dataset by class names.
        
        Args:
            class_names: List of class names to keep
            
        Returns:
            New filtered dataset
        """
        if self.task == "transcription":
            return self
        
        filtered_df = self.metadata_df[
            self.metadata_df[self.label_column].isin(class_names)
        ].copy()
        
        return AudioDataset(
            filtered_df,
            self.root_dir,
            self.config,
            self.task,
            self.preprocessor
        )
