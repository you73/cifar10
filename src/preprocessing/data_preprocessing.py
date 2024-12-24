import cv2
import numpy as np
from skimage.color import rgb2gray
from sklearn.decomposition import PCA

def zca_whitening(images):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    
    flat_images = images.reshape(images.shape[0], -1)
    mean = np.mean(flat_images, axis=0)
    centered_images = flat_images - mean
    sigma = np.cov(centered_images, rowvar=False)
    U, S, _ = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCA_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    whitened = np.dot(centered_images, ZCA_matrix.T)
    whitened_images = whitened.reshape(images.shape)
    
    # Essayer une normalisation différente
    whitened_images = (whitened_images - np.mean(whitened_images, axis=0)) / np.std(whitened_images, axis=0)
    
    # Optionnel : remettre les valeurs dans la plage d'origine si nécessaire
    # min_val = np.min(whitened_images)
    # max_val = np.max(whitened_images)
    # normalized_images = (whitened_images - min_val) / (max_val - min_val)
    
    return whitened_images

def standardize(images):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    mean = np.mean(images, axis=(0,1,2), keepdims=True)
    std = np.std(images, axis=(0,1,2), keepdims=True)
    return (images - mean) / std

def normalize(images):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    return images / 255.0

def convert_to_grayscale(images):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    gray_images = np.array([rgb2gray(image) if image.shape[-1] == 3 else image for image in images])
    return gray_images[..., np.newaxis]


def apply_gaussian_filter(images, kernel_size=(5, 5)):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    return np.array([cv2.GaussianBlur(image, kernel_size, 0) for image in images])

def apply_median_filter(images, kernel_size=3):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    return np.array([cv2.medianBlur((image * 255).astype(np.uint8), kernel_size) for image in images])

def binarize_images(images, threshold=128):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    gray_images = convert_to_grayscale(images)
    binarized = (gray_images > threshold/255.0).astype(np.float32)
    return binarized

def detect_edges(images):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    return np.array([cv2.Canny(image.astype(np.uint8), 100, 200) for image in images])

def erode_images(images, kernel_size=(3, 3)):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    kernel = np.ones(kernel_size, np.uint8)
    return np.array([cv2.erode(image, kernel, iterations=1) for image in images])

def dilate_images(images, kernel_size=(3, 3)):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    kernel = np.ones(kernel_size, np.uint8)
    return np.array([cv2.dilate(image, kernel, iterations=1) for image in images])
