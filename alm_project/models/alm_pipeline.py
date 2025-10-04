"""
Main ALM pipeline that combines transcription, emotion recognition, and cultural context.
"""

import torch
import json
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path

from .transcription import MultiLanguageTranscriptionModel
from .emotion_recognition import EmotionRecognitionModel
from .cultural_context import CulturalContextModel
from ..utils.config import Config


class ALMPipeline:
    """Main ALM pipeline that combines all three models."""
    
    def __init__(
        self,
        config: Config,
        transcription_model_path: Optional[str] = None,
        emotion_model_path: Optional[str] = None,
        context_model_path: Optional[str] = None
    ):
        """Initialize ALM pipeline.
        
        Args:
            config: Configuration object
            transcription_model_path: Path to transcription model
            emotion_model_path: Path to emotion model
            context_model_path: Path to context model
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.transcription_model = None
        self.emotion_model = None
        self.context_model = None
        
        # Load models if paths provided
        if transcription_model_path:
            self.load_transcription_model(transcription_model_path)
        
        if emotion_model_path:
            self.load_emotion_model(emotion_model_path)
        
        if context_model_path:
            self.load_context_model(context_model_path)
        
        self.logger.info("ALM Pipeline initialized")
    
    def load_transcription_model(self, model_path: str) -> None:
        """Load transcription model.
        
        Args:
            model_path: Path to transcription model
        """
        try:
            model_name = self.config.get('models.transcription.model_name', 'facebook/wav2vec2-base-960h')
            device = self.config.get('models.transcription.device', 'cuda')
            
            self.transcription_model = MultiLanguageTranscriptionModel(
                device=device,
                default_language='en'
            )
            
            if Path(model_path).exists():
                self.transcription_model.load_model(model_path)
                self.logger.info(f"Loaded transcription model from {model_path}")
            else:
                self.logger.warning(f"Transcription model path not found: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading transcription model: {e}")
    
    def load_emotion_model(self, model_path: str) -> None:
        """Load emotion recognition model.
        
        Args:
            model_path: Path to emotion model
        """
        try:
            model_name = self.config.get('models.emotion.model_name', 'facebook/wav2vec2-base')
            device = self.config.get('models.emotion.device', 'cuda')
            num_emotions = self.config.get('models.emotion.num_classes', 5)
            emotions = self.config.get('models.emotion.emotions', 
                                    ["anger", "disgust", "fear", "happiness", "sadness"])
            
            self.emotion_model = EmotionRecognitionModel(
                model_name=model_name,
                num_emotions=num_emotions,
                emotions=emotions,
                device=device
            )
            
            if Path(model_path).exists():
                self.emotion_model.load_model(model_path)
                self.logger.info(f"Loaded emotion model from {model_path}")
            else:
                self.logger.warning(f"Emotion model path not found: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading emotion model: {e}")
    
    def load_context_model(self, model_path: str) -> None:
        """Load cultural context model.
        
        Args:
            model_path: Path to context model
        """
        try:
            model_name = self.config.get('models.cultural_context.model_name', 'facebook/wav2vec2-base')
            device = self.config.get('models.cultural_context.device', 'cuda')
            num_contexts = self.config.get('models.cultural_context.num_classes', 8)
            contexts = self.config.get('models.cultural_context.contexts',
                                    ["conversation", "music", "environmental", "religious",
                                     "festival", "speech", "non_speech", "multilingual"])
            
            self.context_model = CulturalContextModel(
                model_name=model_name,
                num_contexts=num_contexts,
                contexts=contexts,
                device=device
            )
            
            if Path(model_path).exists():
                self.context_model.load_model(model_path)
                self.logger.info(f"Loaded context model from {model_path}")
            else:
                self.logger.warning(f"Context model path not found: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading context model: {e}")
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """Process audio file through the complete ALM pipeline.
        
        Args:
            audio_path: Path to audio file
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with transcription, emotion, and cultural context
        """
        result = {
            'file_path': str(audio_path),
            'transcription': '',
            'emotion': '',
            'cultural_context': '',
            'confidence': {}
        }
        
        # Load and preprocess audio
        from ..utils.audio_utils import AudioUtils
        audio_utils = AudioUtils()
        
        try:
            audio, sr = audio_utils.load_audio(str(audio_path), sample_rate=16000)
            audio_tensor = audio_utils.audio_to_tensor(audio)
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Transcription
            if self.transcription_model:
                try:
                    if return_confidence:
                        transcription, conf = self.transcription_model.transcribe(
                            audio_tensor, return_confidence=True
                        )
                        result['transcription'] = transcription[0]
                        result['confidence']['transcription'] = float(conf[0])
                    else:
                        transcription = self.transcription_model.transcribe(audio_tensor)
                        result['transcription'] = transcription[0]
                except Exception as e:
                    self.logger.error(f"Error in transcription: {e}")
            
            # Emotion recognition
            if self.emotion_model:
                try:
                    if return_confidence:
                        emotion, conf = self.emotion_model.predict_emotion(
                            audio_tensor, return_probabilities=True
                        )
                        result['emotion'] = emotion[0]
                        result['confidence']['emotion'] = float(conf[0].max())
                    else:
                        emotion = self.emotion_model.predict_emotion(audio_tensor)
                        result['emotion'] = emotion[0]
                except Exception as e:
                    self.logger.error(f"Error in emotion recognition: {e}")
            
            # Cultural context
            if self.context_model:
                try:
                    if return_confidence:
                        context, conf = self.context_model.predict_context(
                            audio_tensor, return_probabilities=True
                        )
                        result['cultural_context'] = context[0]
                        result['confidence']['cultural_context'] = float(conf[0].max())
                    else:
                        context = self.context_model.predict_context(audio_tensor)
                        result['cultural_context'] = context[0]
                except Exception as e:
                    self.logger.error(f"Error in cultural context: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing audio {audio_path}: {e}")
            result['error'] = str(e)
        
        return result
    
    def process_audio_batch(
        self,
        audio_paths: List[Union[str, Path]],
        return_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """Process multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for audio_path in audio_paths:
            result = self.process_audio(audio_path, return_confidence)
            results.append(result)
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.
        
        Returns:
            Dictionary with pipeline information
        """
        info = {
            'transcription_model': None,
            'emotion_model': None,
            'context_model': None
        }
        
        if self.transcription_model:
            info['transcription_model'] = self.transcription_model.get_model_info()
        
        if self.emotion_model:
            info['emotion_model'] = self.emotion_model.get_model_info()
        
        if self.context_model:
            info['context_model'] = self.context_model.get_model_info()
        
        return info
    
    def save_pipeline(self, save_dir: str) -> None:
        """Save all models in the pipeline.
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.transcription_model:
            self.transcription_model.save_model(str(save_path / "transcription_model.pt"))
        
        if self.emotion_model:
            self.emotion_model.save_model(str(save_path / "emotion_model.pt"))
        
        if self.context_model:
            self.context_model.save_model(str(save_path / "context_model.pt"))
        
        # Save pipeline configuration
        config_path = save_path / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.get_pipeline_info(), f, indent=2)
        
        self.logger.info(f"Pipeline saved to {save_dir}")
    
    def load_pipeline(self, load_dir: str) -> None:
        """Load all models in the pipeline.
        
        Args:
            load_dir: Directory to load models from
        """
        load_path = Path(load_dir)
        
        # Load models
        self.load_transcription_model(str(load_path / "transcription_model.pt"))
        self.load_emotion_model(str(load_path / "emotion_model.pt"))
        self.load_context_model(str(load_path / "context_model.pt"))
        
        self.logger.info(f"Pipeline loaded from {load_dir}")
    
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
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif format.lower() == "csv":
            import pandas as pd
            
            # Flatten results for CSV
            flattened_results = []
            for result in results:
                flat_result = {
                    'file_path': result.get('file_path', ''),
                    'transcription': result.get('transcription', ''),
                    'emotion': result.get('emotion', ''),
                    'cultural_context': result.get('cultural_context', '')
                }
                
                # Add confidence scores if available
                if 'confidence' in result:
                    for key, value in result['confidence'].items():
                        flat_result[f'{key}_confidence'] = value
                
                flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Results exported to {output_path}")
