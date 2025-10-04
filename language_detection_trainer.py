#!/usr/bin/env python3
"""
Train Language Detection Model on Metadata
Creates a trained model for accurate language detection
"""

import os
import pandas as pd
import numpy as np
import librosa
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Tuple, Dict
import logging
from pathlib import Path

# Import our comprehensive language detector
from comprehensive_language_detector import comprehensive_language_detector

logger = logging.getLogger(__name__)

class LanguageDetectionTrainer:
    """Train language detection model on metadata."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.feature_names = []
    
    def prepare_training_data(self, metadata_path: str, max_samples_per_language: int = 50) -> Tuple[List[np.ndarray], List[str]]:
        """Prepare training data from metadata."""
        try:
            # Load metadata
            df = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata with {len(df)} files")
            
            # Group by language (we'll need to infer language from filename or other features)
            training_audio = []
            training_labels = []
            
            # Since the metadata doesn't have explicit language labels, we'll use filename patterns
            language_patterns = {
                'en': ['en', 'english', 'eng'],
                'hi': ['hi', 'hindi', 'hin'],
                'ur': ['ur', 'urdu', 'urd'],
                'zh': ['zh', 'chinese', 'mandarin', 'chi'],
                'ta': ['ta', 'tamil', 'tam'],
                'te': ['te', 'telugu', 'tel'],
                'bn': ['bn', 'bengali', 'bangla', 'ben']
            }
            
            # Process files and infer language
            processed_count = 0
            language_counts = {lang: 0 for lang in language_patterns.keys()}
            
            for idx, row in df.iterrows():
                if processed_count >= max_samples_per_language * len(language_patterns):
                    break
                
                filepath = row['filepath']
                filename = row['filename'].lower()
                
                # Check if file exists
                if not os.path.exists(filepath):
                    continue
                
                # Infer language from filename
                detected_lang = None
                for lang, patterns in language_patterns.items():
                    if any(pattern in filename for pattern in patterns):
                        detected_lang = lang
                        break
                
                # If no pattern match, try to detect from audio
                if detected_lang is None:
                    try:
                        # Load audio and detect language
                        audio, sr = librosa.load(filepath, sr=16000)
                        if len(audio) > 0:
                            detected_lang, _, _ = comprehensive_language_detector.detect_language(audio, sr)
                    except Exception as e:
                        logger.warning(f"Error processing {filepath}: {e}")
                        continue
                
                # Skip if still no language detected
                if detected_lang is None:
                    continue
                
                # Check if we have enough samples for this language
                if language_counts[detected_lang] >= max_samples_per_language:
                    continue
                
                try:
                    # Load audio
                    audio, sr = librosa.load(filepath, sr=16000)
                    if len(audio) > 0:
                        training_audio.append(audio)
                        training_labels.append(detected_lang)
                        language_counts[detected_lang] += 1
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            logger.info(f"Processed {processed_count} files, language counts: {language_counts}")
                
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")
                    continue
            
            logger.info(f"Prepared training data: {len(training_audio)} samples")
            logger.info(f"Language distribution: {language_counts}")
            
            return training_audio, training_labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], []
    
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> List[float]:
        """Extract comprehensive features for language detection."""
        try:
            features = []
            
            # 1. Basic spectral features
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.max(zcr),
                np.min(zcr)
            ])
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.max(spectral_centroids),
                np.min(spectral_centroids)
            ])
            
            # 2. MFCC features (most important for language)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([
                    np.mean(mfccs[i]),
                    np.std(mfccs[i]),
                    np.max(mfccs[i]),
                    np.min(mfccs[i])
                ])
            
            # 3. Delta MFCC features
            delta_mfccs = librosa.feature.delta(mfccs)
            for i in range(13):
                features.extend([
                    np.mean(delta_mfccs[i]),
                    np.std(delta_mfccs[i])
                ])
            
            # 4. Delta-delta MFCC features
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            for i in range(13):
                features.extend([
                    np.mean(delta2_mfccs[i]),
                    np.std(delta2_mfccs[i])
                ])
            
            # 5. Spectral features
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features.extend([
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff)
            ])
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_contrast),
                np.std(spectral_contrast)
            ])
            
            # 6. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([
                np.mean(chroma),
                np.std(chroma)
            ])
            
            # 7. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features.extend([
                np.mean(tonnetz),
                np.std(tonnetz)
            ])
            
            # 8. Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(tempo)
            
            # 9. Onset features
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            onset_rate = len(onset_frames) / (len(audio) / sr) if len(audio) > 0 else 0
            features.append(onset_rate)
            
            # 10. Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features.extend([
                    np.mean(pitch_values),
                    np.std(pitch_values),
                    np.max(pitch_values),
                    np.min(pitch_values)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # 11. Energy features
            rms = librosa.feature.rms(y=audio)[0]
            features.extend([
                np.mean(rms),
                np.std(rms)
            ])
            
            # 12. Mel-frequency spectral features
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            features.extend([
                np.mean(mel_spec),
                np.std(mel_spec)
            ])
            
            # 13. Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            features.extend([
                np.mean(spectral_flatness),
                np.std(spectral_flatness)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return [0.0] * 200  # Default feature vector
    
    def train_model(self, training_audio: List[np.ndarray], training_labels: List[str], test_size: float = 0.2):
        """Train the language detection model."""
        try:
            if len(training_audio) == 0:
                logger.error("No training data available")
                return False
            
            logger.info(f"Training model with {len(training_audio)} samples")
            
            # Extract features
            X = []
            y = []
            
            for i, audio in enumerate(training_audio):
                if i % 10 == 0:
                    logger.info(f"Extracting features for sample {i}/{len(training_audio)}")
                
                features = self.extract_features(audio)
                X.append(features)
                y.append(training_labels[i])
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Labels shape: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            logger.info("Training Random Forest classifier...")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            logger.info("Classification Report:")
            logger.info(classification_report(y_test, y_pred))
            
            # Save model
            joblib.dump(self.model, 'trained_language_model.pkl')
            joblib.dump(self.scaler, 'language_scaler.pkl')
            
            self.is_trained = True
            logger.info("Model trained and saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def predict_language(self, audio: np.ndarray, sr: int = 16000) -> Tuple[str, float, Dict[str, float]]:
        """Predict language using trained model."""
        try:
            if not self.is_trained or self.model is None:
                # Fallback to comprehensive detector
                return comprehensive_language_detector.detect_language(audio, sr)
            
            # Extract features
            features = self.extract_features(audio, sr)
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get confidence
            confidence = np.max(probabilities)
            
            # Create language scores
            language_scores = dict(zip(self.model.classes_, probabilities))
            
            return prediction, confidence, language_scores
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return comprehensive_language_detector.detect_language(audio, sr)
    
    def load_model(self):
        """Load trained model."""
        try:
            if os.path.exists('trained_language_model.pkl') and os.path.exists('language_scaler.pkl'):
                self.model = joblib.load('trained_language_model.pkl')
                self.scaler = joblib.load('language_scaler.pkl')
                self.is_trained = True
                logger.info("Loaded trained language model")
                return True
            else:
                logger.warning("No trained model found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def main():
    """Main training function."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = LanguageDetectionTrainer()
    
    # Prepare training data
    logger.info("Preparing training data...")
    training_audio, training_labels = trainer.prepare_training_data('master_metadata.csv', max_samples_per_language=30)
    
    if len(training_audio) == 0:
        logger.error("No training data available. Exiting.")
        return
    
    # Train model
    logger.info("Training language detection model...")
    success = trainer.train_model(training_audio, training_labels)
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main()
