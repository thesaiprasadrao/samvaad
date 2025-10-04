"""
FastAPI server for ALM inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import tempfile
import os
from pathlib import Path
import json
import uuid
from datetime import datetime

from .inference_engine import InferenceEngine
from ..utils.config import Config
from ..utils.logger import setup_logger


# Pydantic models for request/response
class AudioProcessRequest(BaseModel):
    audio_path: str
    return_confidence: bool = True
    return_metadata: bool = True


class BatchProcessRequest(BaseModel):
    audio_paths: List[str]
    return_confidence: bool = True
    return_metadata: bool = True


class ProcessResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class BatchProcessResponse(BaseModel):
    success: bool
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    total_processing_time: Optional[float] = None


# Initialize FastAPI app
app = FastAPI(
    title="ALM Audio Language Model API",
    description="API for Audio Language Model with transcription, emotion recognition, and cultural context",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_engine = None
logger = None


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup."""
    global inference_engine, logger
    
    # Setup logging
    logger = setup_logger("alm_api", level="INFO")
    logger.info("Starting ALM API server")
    
    # Load configuration
    config = Config()
    
    # Initialize inference engine
    inference_engine = InferenceEngine(config)
    
    logger.info("ALM API server started successfully")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ALM Audio Language Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "process": "/process",
            "process_batch": "/process_batch",
            "upload": "/upload",
            "validate": "/validate"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engine_loaded": inference_engine is not None
    }


@app.get("/info")
async def get_info():
    """Get API and engine information."""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    
    return {
        "api_info": {
            "title": "ALM Audio Language Model API",
            "version": "1.0.0",
            "description": "API for Audio Language Model with transcription, emotion recognition, and cultural context"
        },
        "engine_info": inference_engine.get_engine_info(),
        "supported_formats": inference_engine.get_supported_formats()
    }


@app.post("/process", response_model=ProcessResponse)
async def process_audio(request: AudioProcessRequest):
    """Process single audio file."""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    
    try:
        # Validate audio file
        validation = inference_engine.validate_audio_file(request.audio_path)
        if not validation['valid']:
            return ProcessResponse(
                success=False,
                error=f"Invalid audio file: {', '.join(validation['errors'])}"
            )
        
        # Process audio
        start_time = datetime.now()
        result = inference_engine.process_single_audio(
            request.audio_path,
            return_confidence=request.return_confidence,
            return_metadata=request.return_metadata
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessResponse(
            success=True,
            result=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return ProcessResponse(
            success=False,
            error=str(e)
        )


@app.post("/process_batch", response_model=BatchProcessResponse)
async def process_batch(request: BatchProcessRequest):
    """Process multiple audio files."""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    
    try:
        # Validate all audio files
        validation_errors = []
        for audio_path in request.audio_paths:
            validation = inference_engine.validate_audio_file(audio_path)
            if not validation['valid']:
                validation_errors.append(f"{audio_path}: {', '.join(validation['errors'])}")
        
        if validation_errors:
            return BatchProcessResponse(
                success=False,
                error=f"Invalid audio files: {'; '.join(validation_errors)}"
            )
        
        # Process audio files
        start_time = datetime.now()
        results = inference_engine.process_batch_audio(
            request.audio_paths,
            return_confidence=request.return_confidence,
            return_metadata=request.return_metadata
        )
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchProcessResponse(
            success=True,
            results=results,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return BatchProcessResponse(
            success=False,
            error=str(e)
        )


@app.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    return_confidence: bool = True,
    return_metadata: bool = True
):
    """Upload and process audio file."""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    
    # Check file type
    if not file.filename.lower().endswith(tuple(inference_engine.get_supported_formats())):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported formats: {inference_engine.get_supported_formats()}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Process audio from bytes
        result = inference_engine.process_audio_from_bytes(
            content,
            return_confidence=return_confidence
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate_audio(audio_path: str):
    """Validate audio file."""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    
    try:
        validation = inference_engine.validate_audio_file(audio_path)
        return validation
        
    except Exception as e:
        logger.error(f"Error validating audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export")
async def export_results(
    results: List[Dict[str, Any]],
    format: str = "json",
    filename: Optional[str] = None
):
    """Export results to file."""
    try:
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alm_results_{timestamp}.{format}"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format}', delete=False) as f:
            if format.lower() == "json":
                json.dump(results, f, indent=2)
            elif format.lower() == "csv":
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(f.name, index=False)
            else:
                raise HTTPException(status_code=400, detail="Unsupported format")
            
            temp_path = f.name
        
        # Return file
        return FileResponse(
            temp_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_models_status():
    """Get status of loaded models."""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    
    pipeline_info = inference_engine.get_engine_info()['pipeline_info']
    
    status = {
        "transcription_model": pipeline_info['transcription_model'] is not None,
        "emotion_model": pipeline_info['emotion_model'] is not None,
        "context_model": pipeline_info['context_model'] is not None,
        "all_models_loaded": all([
            pipeline_info['transcription_model'] is not None,
            pipeline_info['emotion_model'] is not None,
            pipeline_info['context_model'] is not None
        ])
    }
    
    return status


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI app with custom configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI app instance
    """
    # This function can be used to create the app with custom configuration
    # For now, we'll use the global app
    return app


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "alm_project.inference.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
