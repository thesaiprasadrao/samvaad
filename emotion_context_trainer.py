#!/usr/bin/env python3
"""
Train Emotion and Context Models
Focused training script for emotion recognition and cultural context
"""

import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Tuple, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmotionContextTrainer:
    """Trainer for emotion and context models using traditional ML."""
    
    def __init__(self):
        self.emotion_scaler = StandardScaler()
        self.context_scaler = StandardScaler()
        self.emotion_model = None
        self.context_model = None
        
        # Emotion labels
        self.emotion_labels = ['anger', 'happiness', 'sadness', 'fear', 'disgust']
        
        # Context labels
        self.context_labels = [
            'speech', 'non-speech', 'music', 'environmental', 'religious', 
            'festival', 'conversation', 'announcement', 'traditional_music', 
            'classical_music', 'folk_music', 'religious_chanting', 
            'festival_sounds', 'traffic', 'nature', 'urban', 'rural', 'ceremonial'
        ]
    
    def prepare_training_data(self, metadata_path: str, max_samples_per_category: int = 100):
        """Prepare training data from metadata."""
        try:
            # Load metadata
            df = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata with {len(df)} files")
            
            # Prepare data
            audio_files = []
            emotion_labels = []
            context_labels = []
            
            # Process files
            processed_count = 0
            emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
            context_counts = {context: 0 for context in self.context_labels}
            
            for idx, row in df.iterrows():
                if processed_count >= max_samples_per_category * len(self.emotion_labels):
                    break
                
                filepath = row['filepath']
                filename = row['filename'].lower()
                
                # Check if file exists
                if not os.path.exists(filepath):
                    continue
                
                # Infer emotion from filename
                emotion = self._infer_emotion_from_filename(filename)
                if emotion is None:
                    continue
                
                # Check if we have enough samples for this emotion
                if emotion_counts[emotion] >= max_samples_per_category:
                    continue
                
                # Infer context
                context = self._infer_context_from_filename(filename)
                
                try:
                    # Load audio to check if it's valid
                    audio, sr = librosa.load(filepath, sr=16000)
                    if len(audio) > 0:
                        audio_files.append(filepath)
                        emotion_labels.append(emotion)
                        context_labels.append(context)
                        emotion_counts[emotion] += 1
                        context_counts[context] += 1
                        processed_count += 1
                        
                        if processed_count % 50 == 0:
                            logger.info(f"Processed {processed_count} files, emotion counts: {emotion_counts}")
                
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")
                    continue
            
            logger.info(f"Prepared training data: {len(audio_files)} samples")
            logger.info(f"Emotion distribution: {emotion_counts}")
            logger.info(f"Context distribution: {context_counts}")
            
            return audio_files, emotion_labels, context_labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], [], []
    
    def _infer_emotion_from_filename(self, filename: str) -> str:
        """Infer emotion from filename."""
        if 'anger' in filename or filename.startswith('a'):
            return 'anger'
        elif 'happiness' in filename or filename.startswith('h'):
            return 'happiness'
        elif 'sadness' in filename or filename.startswith('s'):
            return 'sadness'
        elif 'fear' in filename or filename.startswith('f'):
            return 'fear'
        elif 'disgust' in filename or filename.startswith('d'):
            return 'disgust'
        else:
            return None
    
    def _infer_context_from_filename(self, filename: str) -> str:
        """Infer context from filename."""
        if 'speech' in filename or any(emotion in filename for emotion in self.emotion_labels):
            return 'speech'
        elif 'music' in filename:
            return 'music'
        elif 'traffic' in filename:
            return 'traffic'
        elif 'nature' in filename:
            return 'nature'
        else:
            return 'speech'  # Default to speech
    
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
    
    def train_emotion_model(self, audio_files: List[str], emotion_labels: List[str]):
        """Train emotion recognition model."""
        try:
            logger.info("Training emotion recognition model...")
            
            # Extract features
            X = []
            y = []
            
            for i, audio_file in enumerate(audio_files):
                if i % 20 == 0:
                    logger.info(f"Extracting features for sample {i}/{len(audio_files)}")
                
                try:
                    audio, sr = librosa.load(audio_file, sr=16000)
                    features = self.extract_audio_features(audio, sr)
                    X.append(features)
                    y.append(emotion_labels[i])
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")
                    continue
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Labels shape: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.emotion_scaler.fit_transform(X_train)
            X_test_scaled = self.emotion_scaler.transform(X_test)
            
            # Train model
            logger.info("Training Random Forest classifier for emotion...")
            self.emotion_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.emotion_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.emotion_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Emotion model accuracy: {accuracy:.4f}")
            logger.info("Emotion Classification Report:")
            logger.info(classification_report(y_test, y_pred))
            
            # Save model
            joblib.dump(self.emotion_model, 'trained_emotion_model.pkl')
            joblib.dump(self.emotion_scaler, 'emotion_scaler.pkl')
            
            logger.info("Emotion model trained and saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training emotion model: {e}")
            return False
    
    def train_context_model(self, audio_files: List[str], context_labels: List[str]):
        """Train cultural context model."""
        try:
            logger.info("Training cultural context model...")
            
            # Extract features
            X = []
            y = []
            
            for i, audio_file in enumerate(audio_files):
                if i % 20 == 0:
                    logger.info(f"Extracting features for sample {i}/{len(audio_files)}")
                
                try:
                    audio, sr = librosa.load(audio_file, sr=16000)
                    features = self.extract_audio_features(audio, sr)
                    X.append(features)
                    y.append(context_labels[i])
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")
                    continue
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Labels shape: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.context_scaler.fit_transform(X_train)
            X_test_scaled = self.context_scaler.transform(X_test)
            
            # Train model
            logger.info("Training Random Forest classifier for context...")
            self.context_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.context_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.context_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Context model accuracy: {accuracy:.4f}")
            logger.info("Context Classification Report:")
            logger.info(classification_report(y_test, y_pred))
            
            # Save model
            joblib.dump(self.context_model, 'trained_context_model.pkl')
            joblib.dump(self.context_scaler, 'context_scaler.pkl')
            
            logger.info("Context model trained and saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training context model: {e}")
            return False
    
    def train_all_models(self, metadata_path: str):
        """Train all models."""
        try:
            logger.info("Starting model training...")
            
            # Prepare training data
            audio_files, emotion_labels, context_labels = self.prepare_training_data(metadata_path)
            
            if len(audio_files) == 0:
                logger.error("No training data available")
                return False
            
            # Train emotion model
            emotion_success = self.train_emotion_model(audio_files, emotion_labels)
            
            # Train context model
            context_success = self.train_context_model(audio_files, context_labels)
            
            # Summary
            logger.info("Training Summary:")
            logger.info(f"Emotion Model: {'✓' if emotion_success else '✗'}")
            logger.info(f"Context Model: {'✓' if context_success else '✗'}")
            
            return emotion_success and context_success
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            return False

def main():
    """Main training function."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = EmotionContextTrainer()
    
    # Train all models
    success = trainer.train_all_models('master_metadata.csv')
    
    if success:
        logger.info("All models trained successfully!")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main()
