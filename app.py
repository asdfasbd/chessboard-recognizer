#!/usr/bin/env python3
"""
FastAPI backend for Chess FEN Recognition
Loads model once on startup and serves predictions
"""

import os
import base64
import json
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Import your existing modules
from constants import NN_MODEL_PATH, FEN_CHARS, USE_GRAYSCALE
from chessboard_image import get_chessboard_tiles
from utils import compressed_fen

# Initialize FastAPI app
app = FastAPI(
    title="Chess FEN Recognition API",
    description="Convert chess board images to FEN notation",
    version="1.0.0"
)

# Add CORS middleware (allows frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable (loaded once on startup)
model = None
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

# Request/Response models
class PredictRequest(BaseModel):
    image: str  # Base64 encoded image
    save_history: bool = True

class SquarePrediction(BaseModel):
    square: str  # e.g., "a8", "b7"
    piece: str   # FEN character
    confidence: float

class PredictResponse(BaseModel):
    fen: str
    squares: list[SquarePrediction]
    overall_confidence: float
    lichess_url: str
    history_id: Optional[str] = None

class SaveFinalRequest(BaseModel):
    image: str  # Base64 encoded image
    fen: str    # FEN string


@app.on_event("startup")
async def startup_event():
    """Load model once when server starts"""
    global model
    print("🚀 Starting Chess FEN Recognition Server...")
    print(f"📂 Loading model from {NN_MODEL_PATH}")
    
    # Handle .keras extension
    model_path = NN_MODEL_PATH
    if not (model_path.endswith('.keras') or model_path.endswith('.h5')):
        model_path = model_path.replace('.tf', '.keras')
    
    try:
        model = models.load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"📊 Model expects input shape: {model.input_shape}")
        print(f"🎯 Number of classes: {len(FEN_CHARS)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def predict_tile(tile_img_data) -> tuple[str, float]:
    """Predict piece on a single tile"""
    tile_batch = np.expand_dims(tile_img_data, axis=0)
    predictions = model.predict(tile_batch, verbose=0)
    probabilities = predictions[0]
    
    max_probability = np.max(probabilities)
    predicted_index = np.argmax(probabilities)
    
    return (FEN_CHARS[predicted_index], float(max_probability))


def process_chessboard_image(image: Image.Image) -> tuple[str, list[SquarePrediction], float]:
    """
    Process a chessboard image and return FEN + predictions
    """
    # Save temp file for get_chessboard_tiles (it expects a file path)
    temp_path = "/tmp/chess_temp.png"
    image.save(temp_path)
    
    try:
        # Get 64 tiles from the chessboard
        tiles = get_chessboard_tiles(temp_path, use_grayscale=USE_GRAYSCALE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to detect chessboard: {e}")
    
    # Process each tile
    n_channels = 1 if USE_GRAYSCALE else 3
    predictions = []
    square_predictions = []
    
    # Square names (a8, b8, ..., g1, h1)
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    for i in range(64):
        # Convert tile to tensor
        buf = BytesIO()
        tiles[i].save(buf, format='PNG')
        buf.seek(0)
        
        img_data = tf.image.decode_image(buf.getvalue(), channels=n_channels, expand_animations=False)
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
        img_data = tf.image.resize(img_data, [32, 32])
        
        # Predict
        fen_char, confidence = predict_tile(img_data)
        predictions.append((fen_char, confidence))
        
        # Create square name
        rank_idx = i // 8
        file_idx = i % 8
        square_name = files[file_idx] + ranks[rank_idx]
        
        square_predictions.append(
            SquarePrediction(
                square=square_name,
                piece=fen_char,
                confidence=confidence
            )
        )
    
    # Convert to FEN
    predicted_chars = [p[0] for p in predictions]
    board_array = np.reshape(predicted_chars, [8, 8])
    fen = compressed_fen('/'.join([''.join(row) for row in board_array]))
    
    # Calculate overall confidence
    confidences = [p[1] for p in predictions]
    overall_confidence = float(np.mean(confidences))
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return fen, square_predictions, overall_confidence


def save_to_history(image: Image.Image, fen: str, squares: list[SquarePrediction], confidence: float) -> str:
    """Save prediction to history folder"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history_folder = HISTORY_DIR / timestamp
    history_folder.mkdir(exist_ok=True)
    
    # Save original image
    image.save(history_folder / "original.png")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "fen": fen,
        "overall_confidence": confidence,
        "squares": [s.dict() for s in squares]
    }
    
    with open(history_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return timestamp


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Main prediction endpoint
    Accepts base64 encoded image, returns FEN notation
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode image
        image = base64_to_image(request.image)
        
        # Process chessboard
        fen, squares, confidence = process_chessboard_image(image)
        
        # Generate lichess URL
        lichess_url = f"https://lichess.org/editor/{fen}"
        
        # Save to history if requested
        history_id = None
        if request.save_history:
            history_id = save_to_history(image, fen, squares, confidence)
        
        return PredictResponse(
            fen=fen,
            squares=squares,
            overall_confidence=confidence,
            lichess_url=lichess_url,
            history_id=history_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/history")
async def get_history():
    """Get list of all saved predictions"""
    history_items = []
    
    for folder in sorted(HISTORY_DIR.iterdir(), reverse=True):
        if folder.is_dir():
            metadata_file = folder / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    history_items.append({
                        "id": folder.name,
                        "timestamp": metadata.get("timestamp"),
                        "fen": metadata.get("fen"),
                        "confidence": metadata.get("overall_confidence")
                    })
    
    return {"history": history_items}


@app.get("/history/{history_id}")
async def get_history_item(history_id: str):
    """Get specific history item with image"""
    history_folder = HISTORY_DIR / history_id
    
    if not history_folder.exists():
        raise HTTPException(status_code=404, detail="History item not found")
    
    # Load metadata
    with open(history_folder / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load image
    image = Image.open(history_folder / "original.png")
    image_base64 = image_to_base64(image)
    
    return {
        **metadata,
        "image": image_base64
    }


@app.post("/save-final-board")
async def save_final_board(request: SaveFinalRequest):
    """Save final board image to history/final/ with FEN as filename"""
    FINAL_DIR = Path("history/final")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Decode image
        image = base64_to_image(request.image)
        
        # Create filename from FEN (replace / with -)
        fen_position = request.fen.split(' ')[0]  # Get just position part
        filename = fen_position.replace('/', '-') + '.png'
        
        # Save image
        save_path = FINAL_DIR / filename
        image.save(save_path)
        
        return {
            "success": True,
            "filename": filename,
            "path": str(save_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save final board: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("🏁 Starting FastAPI server...")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🌐 Frontend should connect to: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)