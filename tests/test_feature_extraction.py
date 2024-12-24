import numpy as np
from features.feature_extraction import extract_flatten_features, extract_hog_features, extract_color_histogram, extract_sift_features

def test_extract_flatten_features():
    images = np.random.rand(10, 32, 32, 3)
    features = extract_flatten_features(images)
    assert features.shape == (10, 32*32*3)

def test_extract_hog_features():
    images = np.random.rand(10, 32, 32, 3)
    features = extract_hog_features(images)
    assert features.shape[0] == 10

def test_extract_color_histogram():
    images = np.random.rand(10, 32, 32, 3)
    features = extract_color_histogram(images)
    assert features.shape[0] == 10

def test_extract_sift_features():
    images = np.random.rand(10, 32, 32, 3)
    features = extract_sift_features(images)
    assert features.shape[0] == 10
    assert all(feature.size == 128 * 128 for feature in features)
