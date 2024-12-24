import os
import numpy as np

def save_features(features, filename):
    """
    Save the extracted features to a file.
    
    Args:
    features (numpy array): The features to save.
    filename (str): The filename where to save the features.
    """
    np.save(filename, features)

def load_features(filename):
    """
    Load the extracted features from a file.
    
    Args:
    filename (str): The filename from which to load the features.
    
    Returns:
    numpy array: The loaded features.
    """
    return np.load(filename)
