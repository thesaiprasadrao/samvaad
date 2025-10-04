"""
Cultural context classification model for ALM project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np


class CulturalContextModel(nn.Module):
    """Cultural context classification model using Wav2Vec2 as backbone."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_contexts: int = 8,
        contexts: Optional[List[str]] = None,
        device: str = "cuda",
        freeze_backbone: bool = False
    ):
        """Initialize cultural context model.
        
        Args:
            model_name: HuggingFace model name
            num_contexts: Number of context classes
            contexts: List of context names
            device: Device to run model on
            freeze_backbone: Whether to freeze Wav2Vec2 backbone
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Default contexts including Asian cultural contexts
        if contexts is None:
            contexts = [
                "speech", "non-speech", "music", "environmental", 
                "religious", "festival", "conversation", "announcement",
                "traditional_music", "classical_music", "folk_music",
                "religious_chanting", "festival_sounds", "traffic",
                "nature", "urban", "rural", "ceremonial"
            ]
        
        self.num_contexts = len(contexts)
        
        self.contexts = contexts
        self.context_to_id = {context: i for i, context in enumerate(contexts)}
        self.id_to_context = {i: context for i, context in enumerate(contexts)}
        
        # Load Wav2Vec2 backbone
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get hidden size
        config = Wav2Vec2Config.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Enhanced cultural context classification head with attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Multi-scale feature extraction
        self.conv1d_1 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size=5, padding=2)
        self.conv1d_3 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size=7, padding=3)
        
        # Calculate total feature size: hidden_size + 3 * (hidden_size//2)
        total_feature_size = hidden_size + 3 * (hidden_size // 2)
        
        # Enhanced classification head
        self.context_classifier = nn.Sequential(
            nn.Linear(total_feature_size, 1024),  # Correct input size for multi-scale features
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_contexts)
        )
        
        # Move to device
        self.to(device)
        
        self.logger.info(f"Initialized cultural context model with {num_contexts} contexts")
        self.logger.info(f"Contexts: {contexts}")
    
    def forward(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            audio_inputs: Input audio tensor [batch_size, sequence_length]
            
        Returns:
            Context logits [batch_size, num_contexts]
        """
        # Get Wav2Vec2 features
        outputs = self.backbone(audio_inputs)
        features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply attention mechanism
        attended_features, _ = self.attention(features, features, features)
        attended_features = self.layer_norm(attended_features + features)  # Residual connection
        
        # Multi-scale feature extraction
        features_t = attended_features.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        
        conv1 = F.relu(self.conv1d_1(features_t))  # [batch_size, hidden_size//2, seq_len]
        conv2 = F.relu(self.conv1d_2(features_t))  # [batch_size, hidden_size//2, seq_len]
        conv3 = F.relu(self.conv1d_3(features_t))  # [batch_size, hidden_size//2, seq_len]
        
        # Global average pooling for each scale
        pooled1 = torch.mean(conv1, dim=2)  # [batch_size, hidden_size//2]
        pooled2 = torch.mean(conv2, dim=2)  # [batch_size, hidden_size//2]
        pooled3 = torch.mean(conv3, dim=2)  # [batch_size, hidden_size//2]
        
        # Global average pooling of attended features
        pooled_attended = torch.mean(attended_features, dim=1)  # [batch_size, hidden_size]
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([pooled_attended, pooled1, pooled2, pooled3], dim=1)
        
        # Context classification
        context_logits = self.context_classifier(multi_scale_features)
        
        return context_logits
    
    def predict_context(
        self,
        audio_inputs: torch.Tensor,
        return_probabilities: bool = False
    ) -> List[str]:
        """Predict cultural contexts from audio.
        
        Args:
            audio_inputs: Input audio tensor
            return_probabilities: Whether to return probability scores
            
        Returns:
            List of predicted contexts or (contexts, probabilities)
        """
        with torch.no_grad():
            # Get logits
            logits = self.forward(audio_inputs)
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=-1)
            
            # Get predicted contexts
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_contexts = [self.id_to_context[id.item()] for id in predicted_ids]
            
            if return_probabilities:
                return predicted_contexts, probabilities.cpu().numpy()
            
            return predicted_contexts
    
    def predict_context_file(
        self,
        file_path: str,
        return_probabilities: bool = False
    ) -> str:
        """Predict cultural context from single audio file.
        
        Args:
            file_path: Path to audio file
            return_probabilities: Whether to return probability scores
            
        Returns:
            Predicted context or (context, probabilities)
        """
        # Load and preprocess audio
        from ..utils.audio_utils import AudioUtils
        audio_utils = AudioUtils()
        
        audio, sr = audio_utils.load_audio(file_path, sample_rate=16000)
        audio_tensor = audio_utils.audio_to_tensor(audio)
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        audio_tensor = audio_tensor.to(self.device)
        
        # Predict context
        if return_probabilities:
            contexts, probabilities = self.predict_context(
                audio_tensor, return_probabilities=True
            )
            return contexts[0], probabilities[0]
        else:
            contexts = self.predict_context(audio_tensor)
            return contexts[0]
    
    def get_context_confidence(
        self,
        audio_inputs: torch.Tensor
    ) -> List[float]:
        """Get confidence scores for context predictions.
        
        Args:
            audio_inputs: Input audio tensor
            
        Returns:
            List of confidence scores
        """
        with torch.no_grad():
            logits = self.forward(audio_inputs)
            probabilities = F.softmax(logits, dim=-1)
            max_probabilities = torch.max(probabilities, dim=-1)[0]
            
            return max_probabilities.cpu().numpy().tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'num_contexts': self.num_contexts,
            'contexts': self.contexts,
            'device': self.device,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, save_path: str) -> None:
        """Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_contexts': self.num_contexts,
            'contexts': self.contexts
        }, save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load model from disk.
        
        Args:
            load_path: Path to load model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def train_model(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the cultural context model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save trained model
            
        Returns:
            Dictionary with training history
        """
        # Set up optimizer and loss
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.train()
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.forward(audio)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            val_loss, val_accuracy = self._evaluate(val_dataloader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss / len(train_dataloader))
            history['train_accuracy'].append(100 * train_correct / train_total)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Train Acc: {history['train_accuracy'][-1]:.2f}%, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.2f}%"
            )
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return history
    
    def _evaluate(self, dataloader, criterion) -> Tuple[float, float]:
        """Evaluate model on validation data.
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.forward(audio)
                loss = criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(dataloader), 100 * correct / total
