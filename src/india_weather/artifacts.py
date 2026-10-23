"""Checksums and environment metadata for our locally produced model artifacts."""
import hashlib
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(folder):
    """Call only on trusted artifacts created by this project's training code."""
    entries = {}
    for path in sorted(folder.glob('*.joblib')):
        artifact = joblib.load(path)
        entries[path.name] = {
            'sha256': sha256(path), 'bytes': path.stat().st_size,
            'city': artifact['report']['city'],
            'training_data_sha256': artifact['report']['input_sha256'],
            'feature_names': artifact['feature_names'],
            'prediction_rule': artifact['prediction_rule'],
            'estimator_parameters': artifact['model'].steps[-1][1].get_params(),
        }
    write_json(folder/'manifest.json', {
        'format': 'joblib; load only trusted files',
        'environment': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'models': entries})
