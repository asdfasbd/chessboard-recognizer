import os

# Each FEN char represents of the contents of a chessboard tile
FEN_CHARS = '1RNBQKPrnbqkp'

# Base directory for auto-generated chessboard images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHESSBOARDS_DIR = os.path.join(BASE_DIR, 'images', 'chessboards')

# Try to detect the corners of a chessboard in the image
DETECT_CORNERS = False

# Base directory for 32x32 PNG chessboard squares for
# neural network training and testing
TILES_DIR = os.path.join(BASE_DIR, 'images', 'tiles')

# Use grayscale tile PNGs
USE_GRAYSCALE = True

# Where neural network model/weights are stored
NN_MODEL_PATH = os.path.join(BASE_DIR, 'nn', 'model.keras')