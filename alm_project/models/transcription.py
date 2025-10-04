"""
Enhanced Transcription model for ALM project with multi-language support.
Supports English, Hindi, Urdu, Mandarin, Tamil, Telugu, and Bangla.
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer,
    AutoTokenizer, AutoModelForCTC, pipeline
)
from typing import Dict, Any, Optional, List, Union
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Language-specific model configurations
LANGUAGE_MODELS = {
    'en': 'facebook/wav2vec2-base-960h',
    'hi': 'facebook/wav2vec2-large-xlsr-53-hindi',
    'ur': 'facebook/wav2vec2-large-xlsr-53-urdu', 
    'zh': 'facebook/wav2vec2-large-xlsr-53-chinese-zh-cn',
    'ta': 'facebook/wav2vec2-large-xlsr-53-tamil',
    'te': 'facebook/wav2vec2-large-xlsr-53-telugu',
    'bn': 'facebook/wav2vec2-large-xlsr-53-bengali'
}

class MultiLanguageTranscriptionModel(nn.Module):
    """Enhanced transcription model with multi-language support for Asian languages."""
    
    def __init__(
        self,
        device: str = "cuda",
        default_language: str = "en"
    ):
        """Initialize multi-language transcription model.
        
        Args:
            device: Device to run model on
            default_language: Default language for transcription
        """
        super().__init__()
        
        self.device = device
        self.default_language = default_language
        self.logger = logging.getLogger(__name__)
        
        # Initialize language models
        self.models = {}
        self.processors = {}
        self.language_detector = None
        
        # Load default language model
        self._load_language_model(default_language)
        
        # Initialize language detection
        self._setup_language_detection()
        
        self.logger.info(f"Initialized multi-language transcription model")
        self.logger.info(f"Supported languages: {list(LANGUAGE_MODELS.keys())}")
    
    def _load_language_model(self, language: str) -> None:
        """Load model for specific language.
        
        Args:
            language: Language code (en, hi, ur, zh, ta, te, bn)
        """
        try:
            if language not in LANGUAGE_MODELS:
                self.logger.warning(f"Language {language} not supported, using English")
                language = 'en'
            
            model_name = LANGUAGE_MODELS[language]
            self.logger.info(f"Loading model for {language}: {model_name}")
            
            # Load model and processor
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            
            # Move to device
            model.to(self.device)
            
            self.models[language] = model
            self.processors[language] = processor
            
            self.logger.info(f"Successfully loaded {language} model")
            
        except Exception as e:
            self.logger.error(f"Error loading {language} model: {e}")
            # Fallback to English
            if language != 'en':
                self._load_language_model('en')
    
    def _setup_language_detection(self) -> None:
        """Setup language detection using audio features."""
        # Simple language detection based on audio characteristics
        # In production, you would use a more sophisticated language detection model
        self.language_detector = self._detect_language_simple
    
    def _detect_language_simple(self, audio: np.ndarray, sr: int) -> str:
        """Simple language detection based on audio features.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Detected language code
        """
        try:
            # Extract audio features for language detection
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Simple heuristics for language detection
            # This is a placeholder - in production, use a trained language detection model
            zcr_mean = np.mean(zero_crossing_rate)
            spectral_mean = np.mean(spectral_centroids)
            mfcc_mean = np.mean(mfccs)
            
            # Basic language classification based on audio characteristics
            if zcr_mean > 0.15 and spectral_mean > 2000:
                return 'hi'  # Hindi - typically has more complex phonemes
            elif zcr_mean < 0.08 and spectral_mean < 1500:
                return 'zh'  # Mandarin - tonal language
            elif zcr_mean > 0.12 and spectral_mean > 1800:
                return 'ur'  # Urdu - similar to Hindi
            elif zcr_mean > 0.10 and spectral_mean > 1600:
                return 'ta'  # Tamil - Dravidian language
            elif zcr_mean > 0.11 and spectral_mean > 1700:
                return 'te'  # Telugu - Dravidian language
            elif zcr_mean > 0.13 and spectral_mean > 1900:
                return 'bn'  # Bengali - Indo-Aryan language
            else:
                return 'en'  # English - default
                
        except Exception as e:
            self.logger.error(f"Error in language detection: {e}")
            return self.default_language
    
    def transcribe(
        self,
        audio_inputs: torch.Tensor,
        language: Optional[str] = None,
        return_confidence: bool = False,
        auto_detect_language: bool = True
    ) -> Union[List[str], tuple]:
        """Transcribe audio to text with language support.
        
        Args:
            audio_inputs: Input audio tensor
            language: Target language code (if None, auto-detect)
            return_confidence: Whether to return confidence scores
            auto_detect_language: Whether to auto-detect language
            
        Returns:
            List of transcribed texts or (texts, confidences) if return_confidence=True
        """
        with torch.no_grad():
            # Convert tensor to numpy for language detection
            if hasattr(audio_inputs, 'cpu'):
                audio_np = audio_inputs.cpu().numpy()
            elif hasattr(audio_inputs, 'numpy'):
                audio_np = audio_inputs.numpy()
            else:
                audio_np = audio_inputs
            
            # Auto-detect language if not specified
            if language is None and auto_detect_language:
                language = self._detect_language_simple(audio_np, 16000)
                self.logger.info(f"Auto-detected language: {language}")
            
            # Use default language if detection fails
            if language not in self.models:
                language = self.default_language
                self.logger.warning(f"Language {language} not available, using {self.default_language}")
            
            # Get model and processor
            model = self.models[language]
            processor = self.processors[language]
            
            # Process audio inputs
            if audio_inputs.dim() == 2:
                # Batch of 1D audio samples
                audio_list = [audio_inputs[i].numpy() for i in range(audio_inputs.size(0))]
            else:
                # Single 1D audio sample
                audio_list = [audio_np]
            
            # Process with language-specific processor
            inputs = processor(
                audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            
            if return_confidence:
                # Calculate confidence scores
                probs = torch.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                confidences = torch.mean(max_probs, dim=-1)
                
                return transcriptions, confidences.cpu().numpy()
            
            return transcriptions
    
    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        return_confidence: bool = False,
        auto_detect_language: bool = True
    ) -> Union[str, tuple]:
        """Transcribe single audio file with language support.
        
        Args:
            file_path: Path to audio file
            language: Target language code
            return_confidence: Whether to return confidence score
            auto_detect_language: Whether to auto-detect language
            
        Returns:
            Transcribed text or (text, confidence) if return_confidence=True
        """
        # Load and preprocess audio
        audio, sr = librosa.load(file_path, sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        # Transcribe
        if return_confidence:
            transcriptions, confidences = self.transcribe(
                audio_tensor, 
                language=language,
                return_confidence=True,
                auto_detect_language=auto_detect_language
            )
            return transcriptions[0], confidences[0]
        else:
            transcriptions = self.transcribe(
                audio_tensor,
                language=language,
                auto_detect_language=auto_detect_language
            )
            return transcriptions[0]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return list(self.models.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'supported_languages': self.get_supported_languages(),
            'default_language': self.default_language,
            'device': self.device,
            'num_parameters': sum(
                sum(p.numel() for p in model.parameters()) 
                for model in self.models.values()
            )
        }
    
    def save_model(self, save_path: str) -> None:
        """Save all language models to disk.
        
        Args:
            save_path: Directory to save models
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for language, model in self.models.items():
            model_path = save_path / f"transcription_model_{language}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'language': language,
                'model_name': LANGUAGE_MODELS[language]
            }, model_path)
        
        # Save processor configurations
        processor_info = {}
        for language, processor in self.processors.items():
            processor_info[language] = {
                'vocab_size': len(processor.tokenizer.get_vocab()),
                'model_name': LANGUAGE_MODELS[language]
            }
        
        with open(save_path / "processor_info.json", 'w') as f:
            import json
            json.dump(processor_info, f, indent=2)
        
        self.logger.info(f"Multi-language models saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load all language models from disk.
        
        Args:
            load_path: Directory to load models from
        """
        load_path = Path(load_path)
        
        # Load processor info
        processor_info_path = load_path / "processor_info.json"
        if processor_info_path.exists():
            import json
            with open(processor_info_path, 'r') as f:
                processor_info = json.load(f)
            
            # Load models and processors
            for language, info in processor_info.items():
                model_path = load_path / f"transcription_model_{language}.pt"
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model_name = info['model_name']
                    
                    # Load model and processor
                    model = Wav2Vec2ForCTC.from_pretrained(model_name)
                    processor = Wav2Vec2Processor.from_pretrained(model_name)
                    
                    # Load state dict
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    
                    self.models[language] = model
                    self.processors[language] = processor
                    
                    self.logger.info(f"Loaded {language} model from {model_path}")
        
        self.logger.info(f"Multi-language models loaded from {load_path}")


# Backward compatibility - keep original class name
TranscriptionModel = MultiLanguageTranscriptionModel
