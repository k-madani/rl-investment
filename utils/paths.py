"""
Path management for rl-investment project.

Centralizes all file paths to maintain clean project structure.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Results directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
DATA_DIR = os.path.join(RESULTS_DIR, 'data')

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_figure_path(filename):
    """Get full path for a figure file."""
    return os.path.join(FIGURES_DIR, filename)

def get_model_path(filename):
    """Get full path for a model file."""
    return os.path.join(MODELS_DIR, filename)

def get_data_path(filename):
    """Get full path for a data file."""
    return os.path.join(DATA_DIR, filename)

if __name__ == "__main__":
    print("Project paths:")
    print(f"Root: {PROJECT_ROOT}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Models: {MODELS_DIR}")
    print(f"Data: {DATA_DIR}")