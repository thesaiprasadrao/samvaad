"""
Audio processing utilities for ALM project.
"""

import librosa
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional, Union
import soundfile as sf
from pathlib import Path


class AudioUtils:
    """Utility class for audio processing operations."""
    
    @staticmethod
    def load_audio(
        file_path: Union[str, Path],
        sample_rate: int = 16000,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate
            mono: Convert to mono if True
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(str(file_path), sr=sample_rate, mono=mono)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {e}")
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level.
        
        Args:
            audio: Input audio array
            target_db: Target dB level
            
        Returns:
            Normalized audio array
        """
        # Calculate RMS and target RMS
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(target_db / 20)
        
        if rms > 0:
            return audio * (target_rms / rms)
        return audio
    
    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sample_rate: int,
        top_db: float = 20.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """Trim silence from audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            top_db: Silence threshold in dB
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            Trimmed audio array
        """
        try:
            # Trim silence from beginning and end
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            return trimmed
        except Exception as e:
            # If trimming fails, return original audio
            print(f"Warning: Could not trim silence: {e}")
            return audio
    
    @staticmethod
    def pad_or_truncate(
        audio: np.ndarray,
        target_length: int,
        pad_value: float = 0.0
    ) -> np.ndarray:
        """Pad or truncate audio to target length.
        
        Args:
            audio: Input audio array
            target_length: Target length in samples
            pad_value: Value to use for padding
            
        Returns:
            Padded or truncated audio array
        """
        if len(audio) > target_length:
            # Truncate
            return audio[:target_length]
        elif len(audio) < target_length:
            # Pad
            padding = np.full(target_length - len(audio), pad_value)
            return np.concatenate([audio, padding])
        else:
            return audio
    
    @staticmethod
    def extract_features(
        audio: np.ndarray,
        sample_rate: int,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """Extract MFCC features from audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            MFCC features array
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return mfccs
    
    @staticmethod
    def audio_to_tensor(
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """Convert audio array to PyTorch tensor.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            
        Returns:
            PyTorch tensor
        """
        # Convert to tensor
        tensor = torch.from_numpy(audio).float()
        
        # Ensure it's 1D (sequence_length) for Wav2Vec2
        if tensor.dim() > 1:
            tensor = tensor.squeeze()
        
        return tensor
    
    @staticmethod
    def tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to audio array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Audio array
        """
        # Remove batch dimension if present
        if tensor.dim() > 1:
            tensor = tensor.squeeze()
        
        return tensor.detach().cpu().numpy()
    
    @staticmethod
    def save_audio(
        audio: np.ndarray,
        file_path: Union[str, Path],
        sample_rate: int = 16000
    ) -> None:
        """Save audio array to file.
        
        Args:
            audio: Audio array to save
            file_path: Output file path
            sample_rate: Sample rate
        """
        sf.write(str(file_path), audio, sample_rate)
    
    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> dict:
        """Get audio file information.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            info = sf.info(str(file_path))
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format': info.format
            }
        except Exception as e:
            raise ValueError(f"Error getting audio info for {file_path}: {e}")
    
    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate.
        
        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
        
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
