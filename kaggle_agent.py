#!/usr/bin/env python3
"""
Kaggle-Slaying Multi-Agent Team
Quick launcher script
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run CLI
from cli import main

if __name__ == "__main__":
    main()