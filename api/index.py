import sys
import os

# Add the project root (parent directory of /api) to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
