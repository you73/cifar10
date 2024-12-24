import numpy as np
from preprocessing.data_preprocessing import (
    zca_whitening, standardize, normalize, convert_to_grayscale, 
    apply_gaussian_filter, apply_median_filter, binarize_images, 
    detect_edges, erode_images, dilate_images
)

def test_zca_whitening():
    images = np.random.rand(10, 32, 32, 3)
    whitened_images = zca_whitening(images)
    assert whitened_images.shape == images.shape
    assert not np.array_equal(images, whitened_images)

def test_standardize():
    images = np.random.rand(10, 32, 32, 3)
    standardized_images = standardize(images)
    assert standardized_images.shape == images.shape
    assert np.allclose(np.mean(standardized_images, axis=(0,1,2)), 0, atol=1e-7)
    assert np.allclose(np.std(standardized_images, axis=(0,1,2)), 1, atol=1e-7)

def test_normalize():
    images = np.random.rand(10, 32, 32, 3)
    normalized_images = normalize(images)
    assert normalized_images.shape == images.shape
    assert np.min(normalized_images) >= 0
    assert np.max(normalized_images) <= 1

def test_convert_to_grayscale():
    images = np.random.rand(10, 32, 32, 3)
    grayscale_images = convert_to_grayscale(images)
    expected_grayscale = np.dot(images[...,:3], [0.2989, 0.5870, 0.1140]).reshape(10, 32, 32, 1)
    assert grayscale_images.shape == (10, 32, 32, 1)
    assert np.allclose(grayscale_images, expected_grayscale, atol=0.15)

def test_apply_gaussian_filter():
    images = np.random.rand(10, 32, 32, 3)
    filtered_images = apply_gaussian_filter(images)
    assert filtered_images.shape == images.shape
    assert not np.array_equal(images, filtered_images)

def test_apply_median_filter():
    images = np.random.rand(10, 32, 32, 3)
    filtered_images = apply_median_filter(images)
    assert filtered_images.shape == images.shape
    assert not np.array_equal(images, filtered_images)

def test_binarize_images():
    images = np.random.rand(10, 32, 32, 3)
    binarized_images = binarize_images(images)
    assert binarized_images.shape == (10, 32, 32, 1)
    assert set(np.unique(binarized_images)).issubset({0, 1})

def test_detect_edges():
    images = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
    edges = detect_edges(images)
    assert edges.shape == (10, 32, 32)
    assert np.max(edges) == 255 or np.max(edges) == 1

def test_erode_images():
    images = np.random.rand(10, 32, 32, 3)
    eroded_images = erode_images(images)
    assert eroded_images.shape == images.shape
    assert not np.array_equal(images, eroded_images)

def test_dilate_images():
    images = np.random.rand(10, 32, 32, 3)
    dilated_images = dilate_images(images)
    assert dilated_images.shape == images.shape
    assert not np.array_equal(images, dilated_images)
