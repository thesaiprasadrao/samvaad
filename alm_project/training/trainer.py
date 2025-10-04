"""
Training utilities for ALM models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from pathlib import Path
import time

from ..models.transcription import TranscriptionModel
from ..models.emotion_recognition import EmotionRecognitionModel
from ..models.cultural_context import CulturalContextModel
from ..utils.config import Config


class ModelTrainer:
    """Training utilities for ALM models."""
    
    def __init__(self, config: Config):
        """Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Training configuration
        self.epochs = config.get('training.epochs', 10)
        self.learning_rate = config.get('training.learning_rate', 1e-4)
        self.weight_decay = config.get('training.weight_decay', 1e-5)
        self.patience = config.get('training.patience', 5)
        self.save_dir = config.get('training.save_dir', 'checkpoints')
        self.log_dir = config.get('training.log_dir', 'logs')
        
        # Create directories
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def train_transcription_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Train transcription model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Model name
            device: Device to train on
            
        Returns:
            Training history
        """
        self.logger.info("Starting transcription model training")
        
        # Initialize model
        model = TranscriptionModel(model_name=model_name, device=device)
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_wer': [],
            'best_val_loss': float('inf'),
            'patience_counter': 0
        }
        
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            print(f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                audio = batch['audio'].to(device)
                transcriptions = batch['transcription']
                
                # Forward pass
                logits = model(audio)
                
                # Calculate loss (simplified CTC loss)
                loss = self._calculate_ctc_loss(logits, transcriptions)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation phase
            val_loss, val_wer = self._evaluate_transcription(model, val_loader, device)
            
            # Update history
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_wer'].append(val_wer)
            
            # Check for improvement
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['patience_counter'] = 0
                best_model_state = model.state_dict().copy()
            else:
                history['patience_counter'] += 1
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val WER: {val_wer:.4f}"
            )
            
            # Early stopping
            if history['patience_counter'] >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Save model
        model_path = Path(self.save_dir) / "transcription_model.pt"
        model.save_model(str(model_path))
        
        # Save training history
        history_path = Path(self.log_dir) / "transcription_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def train_emotion_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "facebook/wav2vec2-base",
        num_emotions: int = 5,
        emotions: Optional[List[str]] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Train emotion recognition model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Model name
            num_emotions: Number of emotion classes
            emotions: List of emotion names
            device: Device to train on
            
        Returns:
            Training history
        """
        self.logger.info("Starting emotion recognition model training")
        
        # Initialize model
        model = EmotionRecognitionModel(
            model_name=model_name,
            num_emotions=num_emotions,
            emotions=emotions,
            device=device
        )
        
        # Set up optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'best_val_accuracy': 0.0,
            'patience_counter': 0
        }
        
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            print(f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(audio)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100 * train_correct / train_total:.2f}%")
            
            # Validation phase
            val_loss, val_accuracy = self._evaluate_classification(
                model, val_loader, criterion, device
            )
            
            # Update history
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Check for improvement
            if val_accuracy > history['best_val_accuracy']:
                history['best_val_accuracy'] = val_accuracy
                history['patience_counter'] = 0
                best_model_state = model.state_dict().copy()
            else:
                history['patience_counter'] += 1
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
            
            # Early stopping
            if history['patience_counter'] >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Save model
        model_path = Path(self.save_dir) / "emotion_model.pt"
        model.save_model(str(model_path))
        
        # Save training history
        history_path = Path(self.log_dir) / "emotion_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def train_context_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "facebook/wav2vec2-base",
        num_contexts: int = 8,
        contexts: Optional[List[str]] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Train cultural context model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Model name
            num_contexts: Number of context classes
            contexts: List of context names
            device: Device to train on
            
        Returns:
            Training history
        """
        self.logger.info("Starting cultural context model training")
        
        # Initialize model
        model = CulturalContextModel(
            model_name=model_name,
            num_contexts=num_contexts,
            contexts=contexts,
            device=device
        )
        
        # Set up optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'best_val_accuracy': 0.0,
            'patience_counter': 0
        }
        
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            print(f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(audio)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100 * train_correct / train_total:.2f}%")
            
            # Validation phase
            val_loss, val_accuracy = self._evaluate_classification(
                model, val_loader, criterion, device
            )
            
            # Update history
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Check for improvement
            if val_accuracy > history['best_val_accuracy']:
                history['best_val_accuracy'] = val_accuracy
                history['patience_counter'] = 0
                best_model_state = model.state_dict().copy()
            else:
                history['patience_counter'] += 1
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
            
            # Early stopping
            if history['patience_counter'] >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Save model
        model_path = Path(self.save_dir) / "context_model.pt"
        model.save_model(str(model_path))
        
        # Save training history
        history_path = Path(self.log_dir) / "context_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def _calculate_ctc_loss(
        self,
        logits: torch.Tensor,
        transcriptions: List[str]
    ) -> torch.Tensor:
        """Calculate CTC loss for transcription training.
        
        Args:
            logits: Model logits
            transcriptions: Ground truth transcriptions
            
        Returns:
            CTC loss
        """
        # This is a simplified implementation
        # In practice, you would need to properly encode the transcriptions
        # and use torch.nn.CTCLoss
        
        # For now, return a dummy loss
        return torch.tensor(0.0, requires_grad=True)
    
    def _evaluate_transcription(
        self,
        model: TranscriptionModel,
        dataloader: DataLoader,
        device: str
    ) -> Tuple[float, float]:
        """Evaluate transcription model.
        
        Args:
            model: Transcription model
            dataloader: Validation data loader
            device: Device to evaluate on
            
        Returns:
            Tuple of (average_loss, wer)
        """
        model.eval()
        total_loss = 0.0
        total_wer = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                audio = batch['audio'].to(device)
                transcriptions = batch['transcription']
                
                # Forward pass
                logits = model(audio)
                
                # Calculate loss
                loss = self._calculate_ctc_loss(logits, transcriptions)
                total_loss += loss.item()
                
                # Calculate WER (simplified)
                predicted = model.transcribe(audio)
                wer = self._calculate_wer(predicted, transcriptions)
                total_wer += wer
        
        return total_loss / len(dataloader), total_wer / len(dataloader)
    
    def _evaluate_classification(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str
    ) -> Tuple[float, float]:
        """Evaluate classification model.
        
        Args:
            model: Classification model
            dataloader: Validation data loader
            criterion: Loss function
            device: Device to evaluate on
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(audio)
                loss = criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(dataloader), 100 * correct / total
    
    def _calculate_wer(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate Word Error Rate.
        
        Args:
            predicted: Predicted transcriptions
            ground_truth: Ground truth transcriptions
            
        Returns:
            Word Error Rate
        """
        # Simplified WER calculation
        # In practice, you would use a proper WER implementation
        return 0.0
