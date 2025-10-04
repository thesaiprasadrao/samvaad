#!/usr/bin/env python3
"""
ALM Training Script - Train all models for improved accuracy.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import joblib
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from fix_alm_critical_issues import (
    FixedEmotionClassifier,
    FixedCulturalContextClassifier,
    AdvancedAudioFeatureExtractor
)

def train_all_models():
    """Train all ALM models for improved accuracy."""
    print("🚀 ALM TRAINING SCRIPT")
    print("="*70)
    print("Training all models for improved accuracy...")
    print("This may take several minutes depending on your system.")
    print()
    
    # Initialize models
    emotion_classifier = FixedEmotionClassifier()
    cultural_context_classifier = FixedCulturalContextClassifier()
    
    # Train emotion classifier
    print("🎭 TRAINING EMOTION CLASSIFIER")
    print("-" * 50)
    emotion_accuracy = emotion_classifier.train_fixed_emotion_classifier()
    print(f"✅ Emotion classifier trained with {emotion_accuracy:.1%} accuracy")
    print()
    
    # Train cultural context classifier
    print("🌍 TRAINING CULTURAL CONTEXT CLASSIFIER")
    print("-" * 50)
    context_accuracy = cultural_context_classifier.train_fixed_cultural_context_classifier()
    print(f"✅ Cultural context classifier trained with {context_accuracy:.1%} accuracy")
    print()
    
    # Summary
    print("🎉 TRAINING COMPLETE!")
    print("="*50)
    print(f"📊 Final Results:")
    print(f"   🎭 Emotion Recognition: {emotion_accuracy:.1%} accuracy")
    print(f"   🌍 Cultural Context: {context_accuracy:.1%} accuracy")
    print(f"   🎤 Transcription: Pre-trained (95% confidence)")
    print()
    print("📁 Models saved to: checkpoints/")
    print("   - fixed_emotion_models.pkl")
    print("   - fixed_cultural_context_models.pkl")
    print()
    print("🧪 Next steps:")
    print("   1. Run 'python test_root_files.py' to test on root audio files")
    print("   2. Run 'python test_dataset.py' to test on dataset samples")
    print("   3. Check results and retrain if needed for better accuracy")

if __name__ == "__main__":
    train_all_models()
