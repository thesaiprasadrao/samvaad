"""
Output formatter for ALM system with structured JSON and pretty terminal output.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class ALMOutputFormatter:
    """Formatter for ALM system outputs with JSON and pretty printing."""
    
    def __init__(self):
        """Initialize the formatter."""
        self.emoji_map = {
            'file': '📂',
            'language': '🌐',
            'emotion': '😊',
            'non_speech': '🔊',
            'scene': '🏙️',
            'reasoning': '💡',
            'confidence': '📊',
            'processing_time': '⏱️',
            'audio_info': '🔊',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        }
        
        # Language mapping
        self.language_map = {
            'en': 'English',
            'hi': 'Hindi',
            'unknown': 'Unknown'
        }
        
        # Emotion mapping with emojis
        self.emotion_map = {
            'anger': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happiness': '😊',
            'sadness': '😢',
            'neutral': '😐',
            'stressed': '😰',
            'excited': '🤩'
        }
        
        # Scene mapping with emojis
        self.scene_map = {
            'airport': '✈️',
            'office': '🏢',
            'home': '🏠',
            'street': '🛣️',
            'restaurant': '🍽️',
            'hospital': '🏥',
            'school': '🏫',
            'park': '🌳',
            'unknown': '❓'
        }
    
    def format_output(
        self,
        audio_file: str,
        transcription: str = "",
        emotion: str = "neutral",
        cultural_context: str = "speech",
        non_speech_events: List[str] = None,
        scene: str = "unknown",
        language: str = "en",
        confidence: Dict[str, float] = None,
        processing_time: float = None,
        audio_metadata: Dict[str, Any] = None,
        reasoning: str = None,
        errors: List[str] = None
    ) -> Dict[str, Any]:
        """
        Format ALM output with structured JSON and pretty printing.
        
        Args:
            audio_file: Path to audio file
            transcription: Transcribed text
            emotion: Detected emotion
            cultural_context: Cultural context classification
            non_speech_events: List of non-speech events detected
            scene: Scene classification
            language: Detected language
            confidence: Confidence scores for each component
            processing_time: Time taken for processing
            audio_metadata: Audio file metadata
            reasoning: Reasoning for the classification
            errors: List of any errors encountered
            
        Returns:
            Dictionary with formatted output
        """
        # Clean up inputs
        non_speech_events = non_speech_events or []
        confidence = confidence or {}
        errors = errors or []
        
        # Generate reasoning if not provided
        if not reasoning:
            reasoning = self._generate_reasoning(
                emotion, cultural_context, scene, non_speech_events, language
            )
        
        # Create structured JSON output
        json_output = {
            "audio": Path(audio_file).name,
            "language": language,
            "emotion": emotion,
            "non_speech": non_speech_events,
            "scene": scene,
            "reasoning": reasoning
        }
        
        # Add optional fields if available
        if transcription:
            json_output["transcription"] = transcription
        if confidence:
            json_output["confidence"] = confidence
        if processing_time:
            json_output["processing_time"] = processing_time
        
        # Print pretty output
        self._print_pretty_output(
            audio_file, emotion, non_speech_events, scene, language,
            confidence, processing_time, audio_metadata, reasoning, errors
        )
        
        # Print JSON for debugging
        print(f"\n📋 JSON Output:")
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
        
        return json_output
    
    def _generate_reasoning(
        self,
        emotion: str,
        cultural_context: str,
        scene: str,
        non_speech_events: List[str],
        language: str
    ) -> str:
        """Generate reasoning for the classification."""
        reasoning_parts = []
        
        # Emotion reasoning
        if emotion in ['stressed', 'fear']:
            reasoning_parts.append(f"Person appears {emotion}")
        elif emotion in ['happiness', 'excited']:
            reasoning_parts.append(f"Person seems {emotion}")
        elif emotion in ['sadness', 'anger']:
            reasoning_parts.append(f"Person sounds {emotion}")
        
        # Scene reasoning
        if scene != "unknown":
            reasoning_parts.append(f"Audio context suggests {scene} environment")
        
        # Non-speech events reasoning
        if non_speech_events:
            events_str = ", ".join(non_speech_events)
            reasoning_parts.append(f"Detected sounds: {events_str}")
        
        # Cultural context reasoning
        if cultural_context == "non-speech":
            reasoning_parts.append("Audio contains primarily non-speech content")
        elif cultural_context == "speech":
            reasoning_parts.append("Audio contains clear speech content")
        
        # Language reasoning
        if language != "unknown":
            reasoning_parts.append(f"Language detected as {self.language_map.get(language, language)}")
        
        return ". ".join(reasoning_parts) + "." if reasoning_parts else "Unable to determine context."
    
    def _print_pretty_output(
        self,
        audio_file: str,
        emotion: str,
        non_speech_events: List[str],
        scene: str,
        language: str,
        confidence: Dict[str, float],
        processing_time: float,
        audio_metadata: Dict[str, Any],
        reasoning: str,
        errors: List[str]
    ):
        """Print formatted output to terminal."""
        print(f"\n{'='*60}")
        print(f"🎵 ALM AUDIO ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # File information
        print(f"{self.emoji_map['file']} File: {Path(audio_file).name}")
        
        # Language
        lang_emoji = self.emoji_map['language']
        lang_name = self.language_map.get(language, language.title())
        print(f"{lang_emoji} Language: {lang_name}")
        
        # Emotion
        emotion_emoji = self.emotion_map.get(emotion, '😐')
        print(f"{emotion_emoji} Emotion: {emotion.title()}")
        
        # Non-speech events
        if non_speech_events:
            events_str = ", ".join(non_speech_events)
            print(f"{self.emoji_map['non_speech']} Non-Speech Events: {events_str}")
        else:
            print(f"{self.emoji_map['non_speech']} Non-Speech Events: None detected")
        
        # Scene
        scene_emoji = self.scene_map.get(scene, '❓')
        print(f"{scene_emoji} Scene: {scene.title()}")
        
        # Reasoning
        print(f"{self.emoji_map['reasoning']} Reasoning: {reasoning}")
        
        # Optional information
        if confidence:
            print(f"\n{self.emoji_map['confidence']} Confidence Scores:")
            for component, score in confidence.items():
                print(f"   {component.title()}: {score:.3f}")
        
        if processing_time:
            print(f"\n{self.emoji_map['processing_time']} Processing Time: {processing_time:.2f}s")
        
        if audio_metadata:
            print(f"\n{self.emoji_map['audio_info']} Audio Information:")
            if 'duration' in audio_metadata:
                print(f"   Duration: {audio_metadata['duration']:.2f}s")
            if 'sample_rate' in audio_metadata:
                print(f"   Sample Rate: {audio_metadata['sample_rate']}Hz")
            if 'channels' in audio_metadata:
                print(f"   Channels: {audio_metadata['channels']}")
        
        if errors:
            print(f"\n{self.emoji_map['error']} Errors:")
            for error in errors:
                print(f"   • {error}")
        
        print(f"{'='*60}")
    
    def format_batch_output(
        self,
        results: List[Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format batch processing results.
        
        Args:
            results: List of individual results
            output_file: Optional file to save results
            
        Returns:
            Batch results dictionary
        """
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING RESULTS")
        print(f"{'='*60}")
        
        batch_output = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(results),
            "results": results
        }
        
        # Summary statistics
        emotions = [r.get('emotion', 'unknown') for r in results]
        languages = [r.get('language', 'unknown') for r in results]
        scenes = [r.get('scene', 'unknown') for r in results]
        
        print(f"📁 Total Files Processed: {len(results)}")
        print(f"🌐 Languages Detected: {', '.join(set(languages))}")
        print(f"😊 Emotions Detected: {', '.join(set(emotions))}")
        print(f"🏙️ Scenes Detected: {', '.join(set(scenes))}")
        
        # Individual results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('audio', 'Unknown')}")
            print(f"   {self.emotion_map.get(result.get('emotion', 'neutral'), '😐')} {result.get('emotion', 'unknown').title()}")
            print(f"   {self.scene_map.get(result.get('scene', 'unknown'), '❓')} {result.get('scene', 'unknown').title()}")
            print(f"   {self.language_map.get(result.get('language', 'unknown'), 'Unknown')}")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_output, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        print(f"{'='*60}")
        
        return batch_output
    
    def format_error_output(
        self,
        audio_file: str,
        error_message: str,
        error_type: str = "processing_error"
    ) -> Dict[str, Any]:
        """
        Format error output.
        
        Args:
            audio_file: Path to audio file
            error_message: Error message
            error_type: Type of error
            
        Returns:
            Error output dictionary
        """
        print(f"\n{self.emoji_map['error']} ERROR PROCESSING: {Path(audio_file).name}")
        print(f"Error Type: {error_type}")
        print(f"Message: {error_message}")
        
        error_output = {
            "audio": Path(audio_file).name,
            "error": True,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n📋 Error JSON:")
        print(json.dumps(error_output, indent=2, ensure_ascii=False))
        
        return error_output
