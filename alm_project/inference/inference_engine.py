"""
Inference engine for ALM pipeline.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import time
from datetime import datetime

from ..models.alm_pipeline import ALMPipeline
from ..utils.config import Config
from ..utils.audio_utils import AudioUtils


class InferenceEngine:
    """Inference engine for ALM pipeline."""
    
    def __init__(
        self,
        config: Config,
        model_paths: Optional[Dict[str, str]] = None
    ):
        """Initialize inference engine.
        
        Args:
            config: Configuration object
            model_paths: Dictionary with model paths
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audio_utils = AudioUtils()
        
        # Initialize pipeline
        self.pipeline = ALMPipeline(config)
        
        # Load models if paths provided
        if model_paths:
            if 'transcription' in model_paths:
                self.pipeline.load_transcription_model(model_paths['transcription'])
            
            if 'emotion' in model_paths:
                self.pipeline.load_emotion_model(model_paths['emotion'])
            
            if 'context' in model_paths:
                self.pipeline.load_context_model(model_paths['context'])
        
        self.logger.info("Inference engine initialized")
    
    def process_single_audio(
        self,
        audio_path: Union[str, Path],
        return_confidence: bool = True,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """Process single audio file.
        
        Args:
            audio_path: Path to audio file
            return_confidence: Whether to return confidence scores
            return_metadata: Whether to return audio metadata
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        try:
            # Process audio through pipeline
            result = self.pipeline.process_audio(
                audio_path, 
                return_confidence=return_confidence
            )
            
            # Add processing metadata
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['timestamp'] = datetime.now().isoformat()
            
            # Add audio metadata if requested
            if return_metadata:
                try:
                    audio_info = self.audio_utils.get_audio_info(audio_path)
                    result['audio_metadata'] = audio_info
                except Exception as e:
                    self.logger.warning(f"Could not get audio metadata: {e}")
                    result['audio_metadata'] = {}
            
            # Add pipeline info
            result['pipeline_info'] = self.pipeline.get_pipeline_info()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing audio {audio_path}: {e}")
            return {
                'file_path': str(audio_path),
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def process_batch_audio(
        self,
        audio_paths: List[Union[str, Path]],
        return_confidence: bool = True,
        return_metadata: bool = True,
        max_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """Process multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            return_confidence: Whether to return confidence scores
            return_metadata: Whether to return audio metadata
            max_workers: Maximum number of workers (for future parallel processing)
            
        Returns:
            List of result dictionaries
        """
        self.logger.info(f"Processing batch of {len(audio_paths)} audio files")
        
        results = []
        
        for i, audio_path in enumerate(audio_paths):
            self.logger.info(f"Processing {i+1}/{len(audio_paths)}: {audio_path}")
            
            result = self.process_single_audio(
                audio_path,
                return_confidence=return_confidence,
                return_metadata=return_metadata
            )
            
            results.append(result)
        
        return results
    
    def process_audio_from_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """Process audio from bytes.
        
        Args:
            audio_bytes: Audio data as bytes
            sample_rate: Sample rate of audio
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        try:
            # Convert bytes to audio tensor
            import io
            import soundfile as sf
            
            # Read audio from bytes
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Resample if necessary
            if sr != sample_rate:
                audio = self.audio_utils.resample_audio(audio, sr, sample_rate)
            
            # Convert to tensor
            audio_tensor = self.audio_utils.audio_to_tensor(audio)
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Process through pipeline components
            result = {
                'transcription': '',
                'emotion': '',
                'cultural_context': '',
                'confidence': {}
            }
            
            # Transcription
            if self.pipeline.transcription_model:
                try:
                    if return_confidence:
                        transcription, conf = self.pipeline.transcription_model.transcribe(
                            audio_tensor, return_confidence=True
                        )
                        result['transcription'] = transcription[0]
                        result['confidence']['transcription'] = float(conf[0])
                    else:
                        transcription = self.pipeline.transcription_model.transcribe(audio_tensor)
                        result['transcription'] = transcription[0]
                except Exception as e:
                    self.logger.error(f"Error in transcription: {e}")
            
            # Emotion recognition
            if self.pipeline.emotion_model:
                try:
                    if return_confidence:
                        emotion, conf = self.pipeline.emotion_model.predict_emotion(
                            audio_tensor, return_probabilities=True
                        )
                        result['emotion'] = emotion[0]
                        result['confidence']['emotion'] = float(conf[0].max())
                    else:
                        emotion = self.pipeline.emotion_model.predict_emotion(audio_tensor)
                        result['emotion'] = emotion[0]
                except Exception as e:
                    self.logger.error(f"Error in emotion recognition: {e}")
            
            # Cultural context
            if self.pipeline.context_model:
                try:
                    if return_confidence:
                        context, conf = self.pipeline.context_model.predict_context(
                            audio_tensor, return_probabilities=True
                        )
                        result['cultural_context'] = context[0]
                        result['confidence']['cultural_context'] = float(conf[0].max())
                    else:
                        context = self.pipeline.context_model.predict_context(audio_tensor)
                        result['cultural_context'] = context[0]
                except Exception as e:
                    self.logger.error(f"Error in cultural context: {e}")
            
            # Add processing metadata
            result['processing_time'] = time.time() - start_time
            result['timestamp'] = datetime.now().isoformat()
            result['audio_length'] = len(audio)
            result['sample_rate'] = sample_rate
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing audio bytes: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats.
        
        Returns:
            List of supported audio formats
        """
        return ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    def validate_audio_file(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with validation results
        """
        audio_path = Path(audio_path)
        
        validation = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if file exists
        if not audio_path.exists():
            validation['errors'].append(f"File does not exist: {audio_path}")
            return validation
        
        # Check file extension
        if audio_path.suffix.lower() not in self.get_supported_formats():
            validation['errors'].append(f"Unsupported format: {audio_path.suffix}")
            return validation
        
        # Check file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # 100MB limit
            validation['warnings'].append(f"Large file size: {file_size_mb:.2f}MB")
        
        # Check audio properties
        try:
            audio_info = self.audio_utils.get_audio_info(audio_path)
            
            # Check duration
            if audio_info['duration'] > 300:  # 5 minutes limit
                validation['warnings'].append(f"Long duration: {audio_info['duration']:.2f}s")
            
            if audio_info['duration'] < 0.1:  # 100ms minimum
                validation['errors'].append(f"Too short duration: {audio_info['duration']:.2f}s")
            
            # Check sample rate
            if audio_info['sample_rate'] < 8000:
                validation['warnings'].append(f"Low sample rate: {audio_info['sample_rate']}Hz")
            
        except Exception as e:
            validation['errors'].append(f"Could not read audio file: {e}")
            return validation
        
        # If no errors, file is valid
        if not validation['errors']:
            validation['valid'] = True
        
        return validation
    
    def export_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ) -> None:
        """Export results to file.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
            format: Output format ('json', 'csv')
        """
        self.pipeline.export_results(results, output_path, format)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get inference engine information.
        
        Returns:
            Dictionary with engine information
        """
        return {
            'pipeline_info': self.pipeline.get_pipeline_info(),
            'supported_formats': self.get_supported_formats(),
            'config': self.config.config
        }
