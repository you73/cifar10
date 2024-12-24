from skimage.feature import hog
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern

def extract_flatten_features(images):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    return images.reshape(images.shape[0], -1)
def extract_surf_features(images, hessian_threshold=400):
    if images is None or len(images) == 0:
        raise ValueError("Input images cannot be empty")
    
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    surf_features = []
    
    for image in images:
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        keypoints, descriptors = surf.detectAndCompute(gray_image, None)
        
        if descriptors is not None:
            surf_features.append(descriptors.flatten())
        else:
            surf_features.append(np.zeros(surf.descriptorSize(), dtype=np.float32))
    
    return np.array(surf_features)
def extract_hog_features(images):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    hog_features = []
    for image in images:
        feature, _ = hog(image, visualize=True, block_norm='L2-Hys', channel_axis=-1)
        hog_features.append(feature)
    return np.array(hog_features)

def extract_lbp_features(images, P=8, R=1):
    if images is None or len(images) == 0:
        raise ValueError("Input images cannot be empty")

    lbp_features = []
    for image in images:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        lbp = local_binary_pattern(gray_image, P, R, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        lbp_features.append(hist)
    
    return np.array(lbp_features)

def extract_color_histogram(images, bins=(8, 8, 8)):
    if images is None or images.size == 0:
        raise ValueError("Input images cannot be empty")
    hist_features = []
    for image in images:
        hist = cv2.calcHist([image.astype(np.uint8)], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.append(hist)
    return np.array(hist_features)



def extract_sift_features(images, num_clusters=50):
    if images is None or len(images) == 0:
        raise ValueError("Input images cannot be empty")

    # Fonction pour extraire les descripteurs SIFT d'un ensemble d'images
    def extract_sift_descriptors(images):
        sift = cv2.SIFT_create()
        all_descriptors = []
        for image in images:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        return np.vstack(all_descriptors)
    
    # Fonction pour créer un dictionnaire visuel
    def create_visual_dictionary(descriptors, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(descriptors)
        return kmeans
    
    # Fonction pour représenter les images sous forme d'histogrammes de mots visuels
    def compute_bow_histogram(images, kmeans, num_clusters):
        sift = cv2.SIFT_create()
        histograms = []
        for image in images:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None:
                words = kmeans.predict(descriptors)
                hist, _ = np.histogram(words, bins=np.arange(num_clusters+1))
                histograms.append(hist)
            else:
                histograms.append(np.zeros(num_clusters))
        return np.array(histograms)

    # Extraire les descripteurs SIFT des images
    sift_descriptors = extract_sift_descriptors(images)

    # Créer un dictionnaire visuel (BoVW) avec k-means clustering
    kmeans = create_visual_dictionary(sift_descriptors, num_clusters)

    # Représenter les images sous forme d'histogrammes de mots visuels
    sift_histograms = compute_bow_histogram(images, kmeans, num_clusters)

    return sift_histograms

def extract_orb_features(images, max_descriptors=128):
    if images is None or len(images) == 0:
        raise ValueError("Input images cannot be empty")
    
    orb = cv2.ORB_create()
    orb_features = []

    for image in images:
        # Assurez-vous que l'image est au bon format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convertir en niveau de gris si nécessaire
        if image.ndim == 3 and image.shape[2] == 3:  # Image couleur
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:  # Image déjà en niveaux de gris
            gray = image
        elif image.ndim == 3 and image.shape[2] == 1:  # Image en niveaux de gris avec une dimension supplémentaire
            gray = image[:, :, 0]
        else:
            raise ValueError("Unexpected image shape: {}".format(image.shape))
        
        # Détecter et calculer les descripteurs ORB
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Gérer le cas où aucun descripteur n'est trouvé
        if descriptors is None:
            descriptors = np.zeros((max_descriptors, 32), dtype=np.uint8)
        elif descriptors.shape[0] < max_descriptors:
            descriptors = np.pad(descriptors, ((0, max_descriptors - descriptors.shape[0]), (0, 0)), mode='constant')
        else:
            descriptors = descriptors[:max_descriptors]
        
        # Aplatir les descripteurs pour l'ajouter à la liste des caractéristiques
        orb_features.append(descriptors.flatten())
    
    return np.array(orb_features)
