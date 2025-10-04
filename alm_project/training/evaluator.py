"""
Evaluation utilities for ALM models.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
from pathlib import Path

from ..models.transcription import TranscriptionModel
from ..models.emotion_recognition import EmotionRecognitionModel
from ..models.cultural_context import CulturalContextModel


class ModelEvaluator:
    """Evaluation utilities for ALM models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_transcription_model(
        self,
        model: TranscriptionModel,
        test_loader,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Evaluate transcription model.
        
        Args:
            model: Transcription model
            test_loader: Test data loader
            device: Device to evaluate on
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating transcription model")
        
        model.eval()
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch in test_loader:
                audio = batch['audio'].to(device)
                transcriptions = batch['transcription']
                
                # Get predictions
                predictions = model.transcribe(audio)
                
                all_predictions.extend(predictions)
                all_ground_truth.extend(transcriptions)
        
        # Calculate metrics
        wer = self._calculate_wer(all_predictions, all_ground_truth)
        cer = self._calculate_cer(all_predictions, all_ground_truth)
        
        metrics = {
            'word_error_rate': wer,
            'character_error_rate': cer,
            'num_samples': len(all_predictions)
        }
        
        self.logger.info(f"Transcription evaluation - WER: {wer:.4f}, CER: {cer:.4f}")
        
        return metrics
    
    def evaluate_emotion_model(
        self,
        model: EmotionRecognitionModel,
        test_loader,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Evaluate emotion recognition model.
        
        Args:
            model: Emotion recognition model
            test_loader: Test data loader
            device: Device to evaluate on
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating emotion recognition model")
        
        model.eval()
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch in test_loader:
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Get predictions
                logits = model(audio)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_ground_truth.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_ground_truth, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_ground_truth, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_ground_truth, all_predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_ground_truth, all_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'num_samples': len(all_predictions)
        }
        
        self.logger.info(f"Emotion evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def evaluate_context_model(
        self,
        model: CulturalContextModel,
        test_loader,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Evaluate cultural context model.
        
        Args:
            model: Cultural context model
            test_loader: Test data loader
            device: Device to evaluate on
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating cultural context model")
        
        model.eval()
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch in test_loader:
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Get predictions
                logits = model(audio)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_ground_truth.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_ground_truth, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_ground_truth, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_ground_truth, all_predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_ground_truth, all_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'num_samples': len(all_predictions)
        }
        
        self.logger.info(f"Context evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def evaluate_pipeline(
        self,
        pipeline,
        test_loader,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Evaluate complete ALM pipeline.
        
        Args:
            pipeline: ALM pipeline
            test_loader: Test data loader
            device: Device to evaluate on
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating ALM pipeline")
        
        all_results = []
        
        for batch in test_loader:
            for i in range(len(batch['audio'])):
                audio_path = batch['filepath'][i]
                
                # Process audio through pipeline
                result = pipeline.process_audio(audio_path, return_confidence=True)
                all_results.append(result)
        
        # Calculate overall metrics
        metrics = {
            'total_samples': len(all_results),
            'transcription_available': sum(1 for r in all_results if r.get('transcription')),
            'emotion_available': sum(1 for r in all_results if r.get('emotion')),
            'context_available': sum(1 for r in all_results if r.get('cultural_context'))
        }
        
        # Calculate average confidence scores
        if all_results and 'confidence' in all_results[0]:
            confidences = [r.get('confidence', {}) for r in all_results]
            avg_confidences = {}
            
            for key in ['transcription', 'emotion', 'cultural_context']:
                scores = [c.get(key, 0) for c in confidences if key in c]
                if scores:
                    avg_confidences[key] = np.mean(scores)
            
            metrics['average_confidence'] = avg_confidences
        
        self.logger.info(f"Pipeline evaluation - {metrics['total_samples']} samples processed")
        
        return metrics
    
    def _calculate_wer(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate Word Error Rate.
        
        Args:
            predictions: Predicted transcriptions
            ground_truth: Ground truth transcriptions
            
        Returns:
            Word Error Rate
        """
        # Simplified WER calculation
        # In practice, you would use a proper WER implementation like jiwer
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
        """Calculate Character Error Rate.
        
        Args:
            predictions: Predicted transcriptions
            ground_truth: Ground truth transcriptions
            
        Returns:
            Character Error Rate
        """
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
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
    
    def generate_evaluation_report(
        self,
        transcription_metrics: Optional[Dict[str, Any]] = None,
        emotion_metrics: Optional[Dict[str, Any]] = None,
        context_metrics: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            transcription_metrics: Transcription evaluation metrics
            emotion_metrics: Emotion recognition evaluation metrics
            context_metrics: Cultural context evaluation metrics
            pipeline_metrics: Pipeline evaluation metrics
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'evaluation_summary': {
                'transcription_evaluated': transcription_metrics is not None,
                'emotion_evaluated': emotion_metrics is not None,
                'context_evaluated': context_metrics is not None,
                'pipeline_evaluated': pipeline_metrics is not None
            },
            'transcription_metrics': transcription_metrics,
            'emotion_metrics': emotion_metrics,
            'context_metrics': context_metrics,
            'pipeline_metrics': pipeline_metrics
        }
        
        # Add summary statistics
        if transcription_metrics:
            report['summary'] = {
                'transcription_wer': transcription_metrics.get('word_error_rate', 0),
                'transcription_cer': transcription_metrics.get('character_error_rate', 0)
            }
        
        if emotion_metrics:
            report['summary'].update({
                'emotion_accuracy': emotion_metrics.get('accuracy', 0),
                'emotion_f1': emotion_metrics.get('f1_score', 0)
            })
        
        if context_metrics:
            report['summary'].update({
                'context_accuracy': context_metrics.get('accuracy', 0),
                'context_f1': context_metrics.get('f1_score', 0)
            })
        
        return report
