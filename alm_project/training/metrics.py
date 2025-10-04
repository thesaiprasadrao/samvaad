"""
Metrics calculation utilities for ALM models.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from typing import Dict, Any, List, Tuple, Optional
import logging


class MetricsCalculator:
    """Metrics calculation utilities for ALM models."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            Dictionary with metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'num_samples': len(y_true)
        }
        
        return metrics
    
    def calculate_transcription_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, Any]:
        """Calculate transcription metrics.
        
        Args:
            predictions: Predicted transcriptions
            ground_truth: Ground truth transcriptions
            
        Returns:
            Dictionary with metrics
        """
        # Word Error Rate
        wer = self._calculate_wer(predictions, ground_truth)
        
        # Character Error Rate
        cer = self._calculate_cer(predictions, ground_truth)
        
        # Sentence Error Rate
        ser = self._calculate_ser(predictions, ground_truth)
        
        # Exact Match Rate
        emr = self._calculate_emr(predictions, ground_truth)
        
        metrics = {
            'word_error_rate': wer,
            'character_error_rate': cer,
            'sentence_error_rate': ser,
            'exact_match_rate': emr,
            'num_samples': len(predictions)
        }
        
        return metrics
    
    def calculate_confidence_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        confidence_scores: List[float]
    ) -> Dict[str, Any]:
        """Calculate confidence-based metrics.
        
        Args:
            predictions: Predicted labels
            ground_truth: Ground truth labels
            confidence_scores: Confidence scores
            
        Returns:
            Dictionary with confidence metrics
        """
        # Convert to numpy arrays
        y_true = np.array(ground_truth)
        y_pred = np.array(predictions)
        confidences = np.array(confidence_scores)
        
        # Correct predictions
        correct = (y_true == y_pred)
        
        # Confidence statistics for correct vs incorrect predictions
        correct_confidences = confidences[correct]
        incorrect_confidences = confidences[~correct]
        
        metrics = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'mean_confidence_correct': np.mean(correct_confidences) if len(correct_confidences) > 0 else 0,
            'mean_confidence_incorrect': np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0,
            'confidence_correlation': np.corrcoef(confidences, correct.astype(float))[0, 1],
            'num_samples': len(predictions)
        }
        
        return metrics
    
    def calculate_audio_metrics(
        self,
        audio_files: List[str],
        durations: List[float]
    ) -> Dict[str, Any]:
        """Calculate audio-specific metrics.
        
        Args:
            audio_files: List of audio file paths
            durations: List of audio durations
            
        Returns:
            Dictionary with audio metrics
        """
        durations = np.array(durations)
        
        metrics = {
            'num_files': len(audio_files),
            'total_duration': np.sum(durations),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'median_duration': np.median(durations)
        }
        
        return metrics
    
    def _calculate_wer(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate Word Error Rate."""
        total_words = 0
        total_errors = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_words = pred.split()
            gt_words = gt.split()
            
            total_words += len(gt_words)
            
            # Simple word-level comparison
            errors = abs(len(pred_words) - len(gt_words))
            for p, g in zip(pred_words, gt_words):
                if p.lower() != g.lower():
                    errors += 1
            
            total_errors += errors
        
        return total_errors / total_words if total_words > 0 else 0.0
    
    def _calculate_cer(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate Character Error Rate."""
        total_chars = 0
        total_errors = 0
        
        for pred, gt in zip(predictions, ground_truth):
            total_chars += len(gt)
            
            # Simple character-level comparison
            errors = abs(len(pred) - len(gt))
            for p, g in zip(pred, gt):
                if p.lower() != g.lower():
                    errors += 1
            
            total_errors += errors
        
        return total_errors / total_chars if total_chars > 0 else 0.0
    
    def _calculate_ser(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate Sentence Error Rate."""
        total_sentences = len(predictions)
        incorrect_sentences = 0
        
        for pred, gt in zip(predictions, ground_truth):
            if pred.lower().strip() != gt.lower().strip():
                incorrect_sentences += 1
        
        return incorrect_sentences / total_sentences if total_sentences > 0 else 0.0
    
    def _calculate_emr(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate Exact Match Rate."""
        total_sentences = len(predictions)
        exact_matches = 0
        
        for pred, gt in zip(predictions, ground_truth):
            if pred.lower().strip() == gt.lower().strip():
                exact_matches += 1
        
        return exact_matches / total_sentences if total_sentences > 0 else 0.0
    
    def calculate_model_complexity(
        self,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Calculate model complexity metrics.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with complexity metrics
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': model_size_mb,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
        
        return metrics
    
    def calculate_training_efficiency(
        self,
        training_history: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Calculate training efficiency metrics.
        
        Args:
            training_history: Training history dictionary
            
        Returns:
            Dictionary with efficiency metrics
        """
        metrics = {}
        
        # Training convergence
        if 'train_loss' in training_history:
            train_losses = training_history['train_loss']
            metrics['final_train_loss'] = train_losses[-1]
            metrics['train_loss_improvement'] = train_losses[0] - train_losses[-1]
            metrics['train_loss_volatility'] = np.std(train_losses)
        
        if 'val_loss' in training_history:
            val_losses = training_history['val_loss']
            metrics['final_val_loss'] = val_losses[-1]
            metrics['val_loss_improvement'] = val_losses[0] - val_losses[-1]
            metrics['val_loss_volatility'] = np.std(val_losses)
        
        if 'val_accuracy' in training_history:
            val_accuracies = training_history['val_accuracy']
            metrics['final_val_accuracy'] = val_accuracies[-1]
            metrics['val_accuracy_improvement'] = val_accuracies[-1] - val_accuracies[0]
            metrics['best_val_accuracy'] = max(val_accuracies)
        
        # Training stability
        if 'train_loss' in training_history and 'val_loss' in training_history:
            train_losses = training_history['train_loss']
            val_losses = training_history['val_loss']
            
            # Overfitting indicator
            final_gap = val_losses[-1] - train_losses[-1]
            metrics['overfitting_gap'] = final_gap
            metrics['is_overfitting'] = final_gap > 0.1
        
        return metrics
