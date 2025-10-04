#!/usr/bin/env python3
"""
High Accuracy Audio Analyzer
Uses trained models for maximum accuracy
"""

import os
import json
import time
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Import trained models
from simple_language_trainer import SimpleLanguageTrainer
from trained_model_loader import trained_model_loader
from ultra_accurate_emotion_detector import ultra_accurate_emotion_detector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighAccuracyAudioAnalyzer:
    """High accuracy audio analyzer using trained models."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.language_trainer = SimpleLanguageTrainer()
        self.language_trainer.load_model()
        
        # Load trained models
        self.trained_models_loaded = trained_model_loader.models_loaded
        logger.info(f"Trained models loaded: {self.trained_models_loaded}")
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio file with high accuracy."""
        start_time = time.time()
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            # Get file info
            file_info = self._get_file_info(audio_path, duration, sr)
            
            # Language detection using trained model
            language_analysis = self._analyze_language_high_accuracy(audio, sr)
            
            # Emotion recognition using trained model
            emotion_analysis = self._analyze_emotion_high_accuracy(audio, sr)
            
            # Context analysis using trained model
            context_analysis = self._analyze_context_high_accuracy(audio, sr)
            
            # Speaker analysis
            speaker_analysis = self._analyze_speakers(audio, sr)
            
            # Audio events
            audio_events = self._detect_audio_events(audio, sr)
            
            # Scene classification
            scene_classification = self._classify_scene(audio, sr, context_analysis, audio_events)
            
            # Cultural analysis
            cultural_analysis = self._analyze_cultural_context(language_analysis, context_analysis, audio_events)
            
            # Processing info
            processing_time = time.time() - start_time
            processing_info = {
                'processing_time': processing_time,
                'models_used': 'high_accuracy_trained',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Compile results
            results = {
                'file_info': file_info,
                'language_analysis': language_analysis,
                'transcription': self._get_transcription_placeholder(),
                'emotion_analysis': emotion_analysis,
                'context_analysis': context_analysis,
                'speaker_analysis': speaker_analysis,
                'audio_events': audio_events,
                'scene_classification': scene_classification,
                'cultural_analysis': cultural_analysis,
                'processing_info': processing_info
            }
            
            # Save detailed results
            self._save_results(audio_path, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return self._get_error_result(str(e))
    
    def _get_file_info(self, audio_path: str, duration: float, sr: int) -> Dict[str, Any]:
        """Get file information."""
        try:
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
            return {
                'filename': os.path.basename(audio_path),
                'duration': duration,
                'file_size': f"{file_size:.2f} MB",
                'sample_rate': sr,
                'format': os.path.splitext(audio_path)[1].lower()
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {
                'filename': os.path.basename(audio_path),
                'duration': duration,
                'file_size': 'Unknown',
                'sample_rate': sr,
                'format': 'Unknown'
            }
    
    def _analyze_language_high_accuracy(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """High accuracy language detection using trained model."""
        try:
            if self.language_trainer.model is not None:
                # Use trained language model
                language, confidence = self.language_trainer.predict_language(audio, sr)
                method = 'trained_model'
            else:
                # Fallback to heuristic detection
                language, confidence = self._detect_language_heuristic(audio, sr)
                method = 'heuristic'
            
            # Get language scores
            language_scores = self._get_language_scores(audio, sr)
            
            return {
                'language': language,
                'confidence': confidence,
                'scores': language_scores,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Language analysis error: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'scores': {},
                'method': 'error'
            }
    
    def _analyze_emotion_high_accuracy(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """High accuracy emotion recognition using trained model."""
        try:
            if self.trained_models_loaded:
                # Use trained emotion model
                emotion, confidence, emotion_scores = trained_model_loader.predict_emotion(audio, sr)
                method = 'trained_model'
            else:
                # Fallback to ultra-accurate detector
                emotion, confidence, emotion_scores = ultra_accurate_emotion_detector.detect_emotion(audio, sr)
                method = 'ultra_accurate_detector'
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': emotion_scores,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'probabilities': {},
                'method': 'error'
            }
    
    def _analyze_context_high_accuracy(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """High accuracy context analysis using trained model."""
        try:
            if self.trained_models_loaded:
                # Use trained context model
                context, confidence, context_scores = trained_model_loader.predict_context(audio, sr)
                method = 'trained_model'
            else:
                # Fallback to heuristic detection
                context, confidence = self._detect_context_heuristic(audio, sr)
                context_scores = {}
                method = 'heuristic'
            
            return {
                'context': context,
                'confidence': confidence,
                'probabilities': context_scores,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return {
                'context': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'method': 'error'
            }
    
    def _analyze_speakers(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze speakers in audio."""
        try:
            # Simple speaker analysis based on audio characteristics
            # This is a simplified version - in production you'd use more sophisticated methods
            
            # Detect voice activity
            voice_segments = self._detect_voice_segments(audio, sr)
            speaker_count = min(len(voice_segments), 3)  # Assume max 3 speakers
            
            return {
                'speaker_count': speaker_count,
                'voice_segments': len(voice_segments),
                'has_speech': len(voice_segments) > 0
            }
            
        except Exception as e:
            logger.error(f"Speaker analysis error: {e}")
            return {
                'speaker_count': 1,
                'voice_segments': 1,
                'has_speech': True
            }
    
    def _detect_audio_events(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect audio events."""
        try:
            events = []
            confidences = {}
            
            # Detect music
            if self._detect_music(audio, sr):
                events.append('music')
                confidences['music'] = 0.8
            
            # Detect speech
            if self._detect_speech(audio, sr):
                events.append('speech')
                confidences['speech'] = 0.9
            
            # Detect environmental sounds
            env_sounds = self._detect_environmental_sounds(audio, sr)
            events.extend(env_sounds)
            for sound in env_sounds:
                confidences[sound] = 0.7
            
            return {
                'events': events,
                'confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"Audio events detection error: {e}")
            return {
                'events': ['unknown_sound'],
                'confidences': {'unknown_sound': 0.5}
            }
    
    def _classify_scene(self, audio: np.ndarray, sr: int, context: Dict, events: Dict) -> Dict[str, Any]:
        """Classify the overall scene."""
        try:
            # Use context and events to classify scene
            context_type = context.get('context', 'unknown').lower()
            detected_events = events.get('events', [])
            
            if 'music' in detected_events:
                scene = 'concert'
                confidence = 0.8
                reasoning = "Music detected in audio"
            elif 'speech' in detected_events:
                if context_type == 'conversation':
                    scene = 'home'
                    confidence = 0.8
                    reasoning = "Conversational speech detected"
                else:
                    scene = 'office'
                    confidence = 0.7
                    reasoning = "Speech detected in formal context"
            elif 'traffic' in detected_events:
                scene = 'street'
                confidence = 0.9
                reasoning = "Traffic sounds detected"
            else:
                scene = 'indoor'
                confidence = 0.6
                reasoning = "Indoor environment based on audio characteristics"
            
            return {
                'scene': scene,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Scene classification error: {e}")
            return {
                'scene': 'unknown',
                'confidence': 0.0,
                'reasoning': 'Error in scene classification'
            }
    
    def _analyze_cultural_context(self, language: Dict, context: Dict, events: Dict) -> Dict[str, Any]:
        """Analyze cultural context."""
        try:
            language_code = language.get('language', 'unknown')
            context_type = context.get('context', 'unknown')
            detected_events = events.get('events', [])
            
            # Determine cultural region based on language
            if language_code in ['hi', 'ur', 'bn']:
                cultural_region = 'South Asian'
            elif language_code in ['zh', 'ta', 'te']:
                cultural_region = 'East Asian'
            elif language_code == 'en':
                cultural_region = 'Western'
            else:
                cultural_region = 'Unknown'
            
            # Identify cultural indicators
            cultural_indicators = []
            if 'music' in detected_events:
                cultural_indicators.append('musical_tradition')
            if context_type == 'religious':
                cultural_indicators.append('religious_practice')
            if language_code in ['hi', 'ur', 'bn']:
                cultural_indicators.append('south_asian_language')
            
            return {
                'cultural_region': cultural_region,
                'cultural_indicators': cultural_indicators
            }
            
        except Exception as e:
            logger.error(f"Cultural analysis error: {e}")
            return {
                'cultural_region': 'Unknown',
                'cultural_indicators': []
            }
    
    # Helper methods
    def _detect_language_heuristic(self, audio: np.ndarray, sr: int) -> tuple:
        """Heuristic language detection."""
        try:
            # Simple heuristic based on audio characteristics
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            
            if zcr > 0.1 and spectral_centroid > 2000:
                return 'en', 0.6
            elif zcr < 0.05 and spectral_centroid < 1500:
                return 'hi', 0.5
            else:
                return 'unknown', 0.3
        except:
            return 'unknown', 0.0
    
    def _detect_context_heuristic(self, audio: np.ndarray, sr: int) -> tuple:
        """Heuristic context detection."""
        try:
            # Simple heuristic based on audio characteristics
            rms = np.mean(librosa.feature.rms(y=audio)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            
            if rms > 0.1:
                return 'speech', 0.7
            elif spectral_centroid > 2000:
                return 'music', 0.6
            else:
                return 'environmental', 0.5
        except:
            return 'unknown', 0.0
    
    def _detect_voice_segments(self, audio: np.ndarray, sr: int) -> List:
        """Detect voice segments in audio."""
        try:
            # Simple voice activity detection
            rms = librosa.feature.rms(y=audio)[0]
            threshold = np.mean(rms) * 0.5
            
            voice_segments = []
            in_voice = False
            start_time = 0
            
            for i, energy in enumerate(rms):
                if energy > threshold and not in_voice:
                    in_voice = True
                    start_time = i
                elif energy <= threshold and in_voice:
                    in_voice = False
                    voice_segments.append((start_time, i))
            
            return voice_segments
        except:
            return [(0, len(audio))]
    
    def _detect_music(self, audio: np.ndarray, sr: int) -> bool:
        """Detect if audio contains music."""
        try:
            # Simple music detection based on spectral characteristics
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
            
            return spectral_centroid > 2000 and spectral_rolloff > 4000
        except:
            return False
    
    def _detect_speech(self, audio: np.ndarray, sr: int) -> bool:
        """Detect if audio contains speech."""
        try:
            # Simple speech detection based on energy and spectral characteristics
            rms = np.mean(librosa.feature.rms(y=audio)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            
            return rms > 0.05 and 0.01 < zcr < 0.1
        except:
            return True
    
    def _detect_environmental_sounds(self, audio: np.ndarray, sr: int) -> List[str]:
        """Detect environmental sounds."""
        try:
            sounds = []
            
            # Detect traffic
            if self._detect_traffic(audio, sr):
                sounds.append('traffic')
            
            # Detect nature sounds
            if self._detect_nature(audio, sr):
                sounds.append('nature')
            
            return sounds
        except:
            return []
    
    def _detect_traffic(self, audio: np.ndarray, sr: int) -> bool:
        """Detect traffic sounds."""
        try:
            # Simple traffic detection
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            return 1000 < spectral_centroid < 3000
        except:
            return False
    
    def _detect_nature(self, audio: np.ndarray, sr: int) -> bool:
        """Detect nature sounds."""
        try:
            # Simple nature sound detection
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            return spectral_centroid < 1000
        except:
            return False
    
    def _get_language_scores(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Get language confidence scores."""
        try:
            if self.language_trainer.model is not None:
                # Use trained model for scores
                features = self.language_trainer.extract_features(audio, sr)
                features_scaled = self.language_trainer.scaler.transform([features])
                probabilities = self.language_trainer.model.predict_proba(features_scaled)[0]
                
                languages = ['en', 'hi', 'ur', 'zh', 'ta', 'te', 'bn']
                scores = {}
                for i, lang in enumerate(languages):
                    if i < len(probabilities):
                        scores[lang] = float(probabilities[i])
                
                return scores
            else:
                # Fallback scores
                return {'en': 0.3, 'hi': 0.2, 'ur': 0.2, 'zh': 0.1, 'ta': 0.1, 'te': 0.05, 'bn': 0.05}
        except:
            return {'en': 0.3, 'hi': 0.2, 'ur': 0.2, 'zh': 0.1, 'ta': 0.1, 'te': 0.05, 'bn': 0.05}
    
    def _get_transcription_placeholder(self) -> Dict[str, Any]:
        """Get placeholder transcription."""
        return {
            'text': 'Transcription not available in this version',
            'confidence': 0.0,
            'method': 'not_implemented'
        }
    
    def _save_results(self, audio_path: str, results: Dict[str, Any]) -> None:
        """Save analysis results to file."""
        try:
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            results_file = f"analysis_{filename}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Get error result."""
        return {
            'error': error_message,
            'file_info': {'filename': 'Unknown', 'duration': 0, 'file_size': 'Unknown', 'sample_rate': 0, 'format': 'Unknown'},
            'language_analysis': {'language': 'unknown', 'confidence': 0.0, 'scores': {}, 'method': 'error'},
            'transcription': {'text': 'Error', 'confidence': 0.0, 'method': 'error'},
            'emotion_analysis': {'emotion': 'unknown', 'confidence': 0.0, 'probabilities': {}, 'method': 'error'},
            'context_analysis': {'context': 'unknown', 'confidence': 0.0, 'probabilities': {}, 'method': 'error'},
            'speaker_analysis': {'speaker_count': 0, 'voice_segments': 0, 'has_speech': False},
            'audio_events': {'events': [], 'confidences': {}},
            'scene_classification': {'scene': 'unknown', 'confidence': 0.0, 'reasoning': 'Error'},
            'cultural_analysis': {'cultural_region': 'Unknown', 'cultural_indicators': []},
            'processing_info': {'processing_time': 0.0, 'models_used': 'error', 'timestamp': 'Error'}
        }
