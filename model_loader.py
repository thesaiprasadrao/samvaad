#!/usr/bin/env python3
"""
Trained Model Loader
Loads and uses trained models for emotion and context detection
"""

import os
import joblib
import numpy as np
import librosa
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class TrainedModelLoader:
    """Loader for trained emotion and context models."""
    
    def __init__(self):
        self.emotion_model = None
        self.emotion_scaler = None
        self.context_model = None
        self.context_scaler = None
        self.models_loaded = False
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models."""
        try:
            # Load emotion model (try quick models first, then regular models)
            if os.path.exists('quick_emotion_model.pkl') and os.path.exists('quick_emotion_scaler.pkl'):
                self.emotion_model = joblib.load('quick_emotion_model.pkl')
                self.emotion_scaler = joblib.load('quick_emotion_scaler.pkl')
                logger.info("Loaded quick emotion model")
            elif os.path.exists('trained_emotion_model.pkl') and os.path.exists('emotion_scaler.pkl'):
                self.emotion_model = joblib.load('trained_emotion_model.pkl')
                self.emotion_scaler = joblib.load('emotion_scaler.pkl')
                logger.info("Loaded trained emotion model")
            else:
                logger.warning("No emotion model found")
            
            # Load context model (try quick models first, then regular models)
            if os.path.exists('quick_context_model.pkl') and os.path.exists('quick_context_scaler.pkl'):
                self.context_model = joblib.load('quick_context_model.pkl')
                self.context_scaler = joblib.load('quick_context_scaler.pkl')
                logger.info("Loaded quick context model")
            elif os.path.exists('trained_context_model.pkl') and os.path.exists('context_scaler.pkl'):
                self.context_model = joblib.load('trained_context_model.pkl')
                self.context_scaler = joblib.load('context_scaler.pkl')
                logger.info("Loaded trained context model")
            else:
                logger.warning("No context model found")
            
            self.models_loaded = self.emotion_model is not None and self.context_model is not None
            
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            self.models_loaded = False
    
    def extract_audio_features(self, audio: np.ndarray, sr: int = 16000) -> List[float]:
        """Extract comprehensive audio features."""
        try:
            features = []
            
            # Ensure audio is long enough
            if len(audio) < 1000:
                audio = np.pad(audio, (0, 1000 - len(audio)), mode='constant')
            
            # 1. Basic spectral features
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([
                float(np.mean(zcr)),
                float(np.std(zcr)),
                float(np.max(zcr)),
                float(np.min(zcr))
            ])
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.std(spectral_centroids)),
                float(np.max(spectral_centroids)),
                float(np.min(spectral_centroids))
            ])
            
            # 2. MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([
                    float(np.mean(mfccs[i])),
                    float(np.std(mfccs[i])),
                    float(np.max(mfccs[i])),
                    float(np.min(mfccs[i]))
                ])
            
            # 3. Delta MFCC features
            delta_mfccs = librosa.feature.delta(mfccs)
            for i in range(13):
                features.extend([
                    float(np.mean(delta_mfccs[i])),
                    float(np.std(delta_mfccs[i]))
                ])
            
            # 4. Spectral features
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features.extend([
                float(np.mean(spectral_bandwidth)),
                float(np.std(spectral_bandwidth))
            ])
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features.extend([
                float(np.mean(spectral_rolloff)),
                float(np.std(spectral_rolloff))
            ])
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features.extend([
                float(np.mean(spectral_contrast)),
                float(np.std(spectral_contrast))
            ])
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([
                float(np.mean(chroma)),
                float(np.std(chroma))
            ])
            
            # 6. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features.extend([
                float(np.mean(tonnetz)),
                float(np.std(tonnetz))
            ])
            
            # 7. Tempo and rhythm
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                features.append(float(tempo))
            except:
                features.append(120.0)
            
            # 8. Onset features
            try:
                onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
                onset_rate = len(onset_frames) / (len(audio) / sr) if len(audio) > 0 else 0
                features.append(float(onset_rate))
            except:
                features.append(2.0)
            
            # 9. Pitch features
            try:
                pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                pitch_values = pitches[pitches > 0]
                if len(pitch_values) > 0:
                    features.extend([
                        float(np.mean(pitch_values)),
                        float(np.std(pitch_values)),
                        float(np.max(pitch_values)),
                        float(np.min(pitch_values))
                    ])
                else:
                    features.extend([200.0, 50.0, 300.0, 100.0])
            except:
                features.extend([200.0, 50.0, 300.0, 100.0])
            
            # 10. Energy features
            rms = librosa.feature.rms(y=audio)[0]
            features.extend([
                float(np.mean(rms)),
                float(np.std(rms))
            ])
            
            # 11. Mel-frequency spectral features
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            features.extend([
                float(np.mean(mel_spec)),
                float(np.std(mel_spec))
            ])
            
            # 12. Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            features.extend([
                float(np.mean(spectral_flatness)),
                float(np.std(spectral_flatness))
            ])
            
            # Ensure we have exactly 100 features
            while len(features) < 100:
                features.append(0.0)
            
            return features[:100]
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return [0.0] * 100
    
    def predict_emotion(self, audio: np.ndarray, sr: int = 16000) -> Tuple[str, float, Dict[str, float]]:
        """Predict emotion using trained model."""
        try:
            if not self.models_loaded or self.emotion_model is None:
                return 'neutral', 0.5, {'neutral': 0.5}
            
            # Extract features
            features = self.extract_audio_features(audio, sr)
            features_scaled = self.emotion_scaler.transform([features])
            
            # Predict
            prediction = self.emotion_model.predict(features_scaled)[0]
            probabilities = self.emotion_model.predict_proba(features_scaled)[0]
            
            # Get confidence
            confidence = np.max(probabilities)
            
            # Create emotion scores
            emotion_scores = dict(zip(self.emotion_model.classes_, probabilities))
            
            return prediction, confidence, emotion_scores
            
        except Exception as e:
            logger.error(f"Emotion prediction error: {e}")
            return 'neutral', 0.5, {'neutral': 0.5}
    
    def predict_context(self, audio: np.ndarray, sr: int = 16000) -> Tuple[str, float, Dict[str, float]]:
        """Predict context using trained model."""
        try:
            if not self.models_loaded or self.context_model is None:
                return 'speech', 0.5, {'speech': 0.5}
            
            # Extract features
            features = self.extract_audio_features(audio, sr)
            features_scaled = self.context_scaler.transform([features])
            
            # Predict
            prediction = self.context_model.predict(features_scaled)[0]
            probabilities = self.context_model.predict_proba(features_scaled)[0]
            
            # Get confidence
            confidence = np.max(probabilities)
            
            # Create context scores
            context_scores = dict(zip(self.context_model.classes_, probabilities))
            
            return prediction, confidence, context_scores
            
        except Exception as e:
            logger.error(f"Context prediction error: {e}")
            return 'speech', 0.5, {'speech': 0.5}

# Global instance
trained_model_loader = TrainedModelLoader()
