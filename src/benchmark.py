import sys
import os
import argparse
from src.utils.io_utils import load_features, save_features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import load_cifar10
from src.features.feature_extraction import extract_flatten_features, extract_hog_features, extract_color_histogram, extract_lbp_features, extract_orb_features, extract_sift_features
from src.models.model_training import train_model

from src.models.model_evaluation import evaluate_model
from src.visualization.visualization_tools import (
    plot_confusion_matrix, plot_roc_curve, plot_learning_curve, plot_decision_boundary, plot_latent_space, plot_cost_function
)
from src.preprocessing.data_preprocessing import zca_whitening, standardize, normalize, convert_to_grayscale, apply_gaussian_filter, apply_median_filter, binarize_images, detect_edges, erode_images, dilate_images
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import learning_curve
import numpy as np

class CIFAR10Benchmark:
    def __init__(self, preprocessor=None, feature_extractor=None, classifier=None, batch_size=32, learning_rate=0.01, kernel='linear'):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.preprocessors = {
            "zca_whitening": zca_whitening,
            "standardize": standardize,
            "normalize": normalize,
            "grayscale": convert_to_grayscale,
            "gaussian_filter": apply_gaussian_filter,
            "median_filter": apply_median_filter,
            "binarize": binarize_images,
            "detect_edges": detect_edges,
            "erode": erode_images,
            "dilate": dilate_images
        }
        self.feature_extractors = {
            "flatten": extract_flatten_features,
            "hog": extract_hog_features,
            "color_histogram": extract_color_histogram,
            "sift": extract_sift_features,
            "orb": extract_orb_features,
            "lbp": extract_lbp_features
        }
        self.classifiers = {
            "sgd": 'sgd',
            "logistic_regression": 'logistic_regression',
            "knn": 'knn',
            "random_forest": 'random_forest',
            "svm": 'svm',
            "decision_tree": 'decision_tree'
        }

    def load_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = load_cifar10()

    def preprocess_data(self):
        if self.preprocessor in self.preprocessors:
            preprocessor = self.preprocessors[self.preprocessor]
            self.x_train = preprocessor(self.x_train)
            self.x_test = preprocessor(self.x_test)

    def extract_features(self):
        if self.feature_extractor in self.feature_extractors:
            feature_extractor = self.feature_extractors[self.feature_extractor]
            features_dir = "data/features"
            os.makedirs(features_dir, exist_ok=True)
            
            # Inclure des détails sur le prétraitement dans le nom du fichier
            preprocessor_suffix = f"{self.preprocessor}_" if self.preprocessor else ""
            features_train_file = os.path.join(features_dir, f"{preprocessor_suffix}{self.feature_extractor}_train.npy")
            features_test_file = os.path.join(features_dir, f"{preprocessor_suffix}{self.feature_extractor}_test.npy")

            if os.path.exists(features_train_file) and os.path.exists(features_test_file):
                self.X_train = load_features(features_train_file)
                self.X_test = load_features(features_test_file)
                print(f"Loaded features from {features_train_file} and {features_test_file}")
            else:
                self.X_train = feature_extractor(self.x_train)
                self.X_test = feature_extractor(self.x_test)
                save_features(self.X_train, features_train_file)
                save_features(self.X_test, features_test_file)
                print(f"Extracted and saved features to {features_train_file} and {features_test_file}")

    def train_model(self):
        if self.classifier in self.classifiers:
            self.model = train_model(
                self.classifier, 
                self.X_train, 
                self.y_train, 
                batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                kernel=self.kernel, 
                preprocessor=self.preprocessor, 
                feature_extractor=self.feature_extractor,
                search='random'
            )

    def evaluate_model(self):
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        self.y_pred = metrics['y_pred']
        self.accuracy = metrics['accuracy']
        self.precision = metrics['precision']
        self.recall = metrics['recall']
        self.f1_score = metrics['f1_score']
        self.report = metrics['classification_report']
        self.roc_auc = metrics['roc_auc']

        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1 Score: {self.f1_score}")
        if self.roc_auc:
            print(f"ROC AUC: {self.roc_auc}")
        print("Classification Report:")
        print(self.report)

    def plot_results(self):
        plot_confusion_matrix(self.y_test, self.y_pred, classes=[str(i) for i in range(10)])
        if self.roc_auc:
            plot_roc_curve(self.y_test, self.model.predict_proba(self.X_test), n_classes=10)
        tsne = TSNE(n_components=2, random_state=42)
        X_test_tsne = tsne.fit_transform(self.X_test)
        plot_latent_space(X_test_tsne, self.y_test)
        if hasattr(self.model, 'loss_curve_'):
            plot_cost_function(self.model.loss_curve_)

    def analyze_features_independence(self):
        correlation_matrix = np.corrcoef(self.X_train, rowvar=False)
        print("Correlation matrix:\n", correlation_matrix)

    def run_benchmark(self):
        print("Benchmarking CIFAR-10 dataset with:")
        print(f"Preprocessor: {self.preprocessor}")
        print(f"Feature Extractor: {self.feature_extractor}")
        print(f"Classifier: {self.classifier}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Kernel: {self.kernel}")

        self.load_data()
        print("Data loaded.")

        self.preprocess_data()
        print("Data preprocessed.")
        self.extract_features()
        print("Features extracted.")
        self.train_model()
        print("Model trained.")
        self.evaluate_model()
        print("Model evaluated.")
        self.plot_results()
        print("Results plotted.")
        self.analyze_features_independence()
        print("Features independence analyzed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CIFAR-10 Benchmark')
    parser.add_argument('--preprocessor', type=str, required=True, help='Preprocessing method')
    parser.add_argument('--feature-extractor', type=str, required=True, help='Feature extraction method')
    parser.add_argument('--classifier', type=str, required=True, help='Classifier')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel type (for SVM)')

    args = parser.parse_args()

    benchmark = CIFAR10Benchmark(
        preprocessor=args.preprocessor,
        feature_extractor=args.feature_extractor,
        classifier=args.classifier,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kernel=args.kernel
    )
    benchmark.run_benchmark()
