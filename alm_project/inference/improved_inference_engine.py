"""
Improved inference engine with better output formatting and error handling.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import time
from datetime import datetime
import librosa
import soundfile as sf

from ..models.transcription import TranscriptionModel
from ..models.emotion_recognition import EmotionRecognitionModel
from ..models.cultural_context import CulturalContextModel
from ..utils.config import Config
from ..utils.audio_utils import AudioUtils
from ..utils.output_formatter import ALMOutputFormatter


class ImprovedInferenceEngine:
    """Improved inference engine with better output formatting."""
    
    def __init__(
        self,
        config: Config,
        model_paths: Optional[Dict[str, str]] = None,
        use_pretrained_transcription: bool = True
    ):
        """Initialize improved inference engine.
        
        Args:
            config: Configuration object
            model_paths: Dictionary with model paths
            use_pretrained_transcription: Whether to use pre-trained transcription
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audio_utils = AudioUtils()
        self.output_formatter = ALMOutputFormatter()
        self.use_pretrained_transcription = use_pretrained_transcription
        
        # Initialize models
        self.transcription_model = None
        self.emotion_model = None
        self.context_model = None
        
        # Load models if paths provided
        if model_paths:
            self._load_models(model_paths)
        
        self.logger.info("Improved inference engine initialized")
    
    def _load_models(self, model_paths: Dict[str, str]):
        """Load all models."""
        try:
            # Load transcription model
            if 'transcription' in model_paths and Path(model_paths['transcription']).exists():
                if self.use_pretrained_transcription:
                    self.logger.info("Using pre-trained transcription (no model loading needed)")
                else:
                    self.transcription_model = TranscriptionModel(
                        model_name="facebook/wav2vec2-base-960h",
                        device="cpu"
                    )
                    self.transcription_model.load_model(model_paths['transcription'])
                    self.logger.info("Transcription model loaded")
            
            # Load emotion model
            if 'emotion' in model_paths and Path(model_paths['emotion']).exists():
                self.emotion_model = EmotionRecognitionModel(
                    model_name="facebook/wav2vec2-base-960h",
                    num_emotions=5,
                    device="cpu"
                )
                self.emotion_model.load_model(model_paths['emotion'])
                self.logger.info("Emotion model loaded")
            
            # Load cultural context model
            if 'context' in model_paths and Path(model_paths['context']).exists():
                self.context_model = CulturalContextModel(
                    model_name="facebook/wav2vec2-base-960h",
                    num_contexts=2,
                    device="cpu"
                )
                self.context_model.load_model(model_paths['context'])
                self.logger.info("Cultural context model loaded")
                
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        return_confidence: bool = True,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """Process single audio file with improved output formatting.
        
        Args:
            audio_path: Path to audio file
            return_confidence: Whether to return confidence scores
            return_metadata: Whether to return audio metadata
            
        Returns:
            Formatted result dictionary
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        errors = []
        
        try:
            # Validate audio file
            if not audio_path.exists():
                return self.output_formatter.format_error_output(
                    str(audio_path), "Audio file does not exist", "file_not_found"
                )
            
            # Load audio
            try:
                audio, sr = librosa.load(str(audio_path), sr=16000)
                duration = len(audio) / sr
            except Exception as e:
                return self.output_formatter.format_error_output(
                    str(audio_path), f"Error loading audio: {e}", "audio_loading_error"
                )
            
            # Initialize results
            transcription = ""
            emotion = "neutral"
            cultural_context = "speech"
            non_speech_events = []
            scene = "unknown"
            language = "en"
            confidence = {}
            
            # Get audio metadata
            audio_metadata = {}
            if return_metadata:
                try:
                    info = sf.info(str(audio_path))
                    audio_metadata = {
                        'duration': duration,
                        'sample_rate': info.samplerate,
                        'channels': info.channels,
                        'format': info.format
                    }
                except Exception as e:
                    errors.append(f"Could not get audio metadata: {e}")
            
            # Process with models
            try:
                # Transcription
                if self.use_pretrained_transcription:
                    transcription = self._transcribe_with_whisper(audio_path)
                elif self.transcription_model:
                    transcription = self._transcribe_with_model(audio)
                
                # Emotion recognition
                if self.emotion_model:
                    emotion_result = self._predict_emotion(audio)
                    emotion = emotion_result.get('emotion', 'neutral')
                    if return_confidence:
                        confidence['emotion'] = emotion_result.get('confidence', 0.0)
                
                # Cultural context
                if self.context_model:
                    context_result = self._predict_context(audio)
                    cultural_context = context_result.get('context', 'speech')
                    if return_confidence:
                        confidence['cultural_context'] = context_result.get('confidence', 0.0)
                
                # Scene classification (simple heuristic)
                scene = self._classify_scene(audio, cultural_context, non_speech_events)
                
                # Language detection (simple heuristic)
                language = self._detect_language(transcription, audio)
                
                # Non-speech events detection (simple heuristic)
                non_speech_events = self._detect_non_speech_events(audio, cultural_context)
                
            except Exception as e:
                errors.append(f"Model processing error: {e}")
                self.logger.error(f"Error in model processing: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            if return_confidence:
                confidence['processing_time'] = processing_time
            
            # Format and return output
            return self.output_formatter.format_output(
                audio_file=str(audio_path),
                transcription=transcription,
                emotion=emotion,
                cultural_context=cultural_context,
                non_speech_events=non_speech_events,
                scene=scene,
                language=language,
                confidence=confidence if return_confidence else None,
                processing_time=processing_time,
                audio_metadata=audio_metadata,
                errors=errors
            )
            
        except Exception as e:
            return self.output_formatter.format_error_output(
                str(audio_path), f"Unexpected error: {e}", "unexpected_error"
            )
    
    def _transcribe_with_whisper(self, audio_path: Path) -> str:
        """Transcribe using Whisper (if available)."""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path))
            return result["text"].strip()
        except ImportError:
            return "Whisper not available"
        except Exception as e:
            self.logger.error(f"Whisper transcription error: {e}")
            return "Transcription failed"
    
    def _transcribe_with_model(self, audio: torch.Tensor) -> str:
        """Transcribe using our trained model."""
        try:
            if self.transcription_model:
                with torch.no_grad():
                    logits = self.transcription_model(audio.unsqueeze(0))
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.transcription_model.processor.decode(predicted_ids[0])
                    return transcription.strip()
        except Exception as e:
            self.logger.error(f"Model transcription error: {e}")
        return "Transcription failed"
    
    def _predict_emotion(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Predict emotion from audio."""
        try:
            if self.emotion_model:
                with torch.no_grad():
                    logits = self.emotion_model(audio.unsqueeze(0))
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness']
                    emotion = emotion_labels[predicted_class] if predicted_class < len(emotion_labels) else 'neutral'
                    
                    return {'emotion': emotion, 'confidence': confidence}
        except Exception as e:
            self.logger.error(f"Emotion prediction error: {e}")
        return {'emotion': 'neutral', 'confidence': 0.0}
    
    def _predict_context(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Predict cultural context from audio."""
        try:
            if self.context_model:
                with torch.no_grad():
                    logits = self.context_model(audio.unsqueeze(0))
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    context_labels = ['speech', 'non-speech']
                    context = context_labels[predicted_class] if predicted_class < len(context_labels) else 'speech'
                    
                    return {'context': context, 'confidence': confidence}
        except Exception as e:
            self.logger.error(f"Context prediction error: {e}")
        return {'context': 'speech', 'confidence': 0.0}
    
    def _classify_scene(self, audio: torch.Tensor, cultural_context: str, non_speech_events: List[str]) -> str:
        """Classify scene using simple heuristics."""
        try:
            # Analyze audio features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio.numpy(), sr=16000)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio.numpy(), sr=16000)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio.numpy())[0]
            
            avg_centroid = spectral_centroids.mean()
            avg_rolloff = spectral_rolloff.mean()
            avg_zcr = zero_crossing_rate.mean()
            
            # Simple scene classification
            if cultural_context == "non-speech":
                if avg_centroid > 2000 and avg_rolloff > 4000:
                    return "music"
                elif avg_zcr > 0.1:
                    return "street"
                else:
                    return "unknown"
            else:
                # Speech audio - try to classify environment
                if avg_centroid > 1500 and avg_rolloff > 2500:
                    return "office"
                elif avg_zcr > 0.05:
                    return "street"
                else:
                    return "home"
                    
        except Exception as e:
            self.logger.error(f"Scene classification error: {e}")
        return "unknown"
    
    def _detect_language(self, transcription: str, audio: torch.Tensor) -> str:
        """Detect language using simple heuristics."""
        if not transcription or transcription == "Transcription failed":
            return "unknown"
        
        # Simple language detection based on characters
        if any(char in transcription for char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'):
            return "hi"
        else:
            return "en"
    
    def _detect_non_speech_events(self, audio: torch.Tensor, cultural_context: str) -> List[str]:
        """Detect non-speech events using simple heuristics."""
        events = []
        
        try:
            # Analyze audio features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio.numpy(), sr=16000)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio.numpy(), sr=16000)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio.numpy())[0]
            
            avg_centroid = spectral_centroids.mean()
            avg_rolloff = spectral_rolloff.mean()
            avg_zcr = zero_crossing_rate.mean()
            
            # Simple event detection
            if cultural_context == "non-speech":
                if avg_centroid > 2000 and avg_rolloff > 4000:
                    events.append("music")
                elif avg_zcr > 0.1:
                    events.append("traffic")
                elif avg_centroid < 1000:
                    events.append("ambient")
                else:
                    events.append("unknown_sound")
            
        except Exception as e:
            self.logger.error(f"Non-speech event detection error: {e}")
        
        return events
    
    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        return_confidence: bool = True,
        return_metadata: bool = True,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process multiple audio files."""
        results = []
        
        for audio_path in audio_paths:
            result = self.process_audio(
                audio_path,
                return_confidence=return_confidence,
                return_metadata=return_metadata
            )
            results.append(result)
        
        return self.output_formatter.format_batch_output(results, output_file)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return {
            'models_loaded': {
                'transcription': self.transcription_model is not None,
                'emotion': self.emotion_model is not None,
                'context': self.context_model is not None
            },
            'use_pretrained_transcription': self.use_pretrained_transcription,
            'supported_formats': ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        }
