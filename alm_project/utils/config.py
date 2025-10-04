"""
Configuration management for ALM project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json


class Config:
    """Configuration manager for ALM project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "root_dir": ".",
                "metadata_file": "master_metadata.csv",
                "sample_rate": 16000,
                "max_duration": 10.0,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15
            },
            "models": {
                "transcription": {
                    "model_name": "facebook/wav2vec2-base-960h",
                    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                    "batch_size": 8,
                    "max_length": 512
                },
                "emotion": {
                    "model_name": "facebook/wav2vec2-base",
                    "num_classes": 5,
                    "emotions": ["anger", "disgust", "fear", "happiness", "sadness"],
                    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                    "batch_size": 16
                },
                "cultural_context": {
                    "model_name": "facebook/wav2vec2-base",
                    "num_classes": 8,
                    "contexts": ["conversation", "music", "environmental", "religious", 
                                "festival", "speech", "non_speech", "multilingual"],
                    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                    "batch_size": 16
                }
            },
            "training": {
                "epochs": 10,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "patience": 5,
                "save_dir": "checkpoints",
                "log_dir": "logs"
            },
            "inference": {
                "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                "batch_size": 1,
                "output_format": "json"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'data.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses current config_path.
        """
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
