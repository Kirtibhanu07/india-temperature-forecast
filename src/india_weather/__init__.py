"""Observed-station temperature forecasting for three Indian cities."""
import os

# Set defaults before importing NumPy/scikit-learn, including prediction entrypoints.
# Users can override these environment variables before starting Python.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

__version__ = "0.1.0"
