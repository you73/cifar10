import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from visualization.visualization_tools import plot_confusion_matrix, plot_roc_curve, plot_learning_curve, plot_decision_boundary, plot_latent_space

def test_plot_confusion_matrix():
    y_test = np.random.randint(0, 10, 100)
    y_pred = np.random.randint(0, 10, 100)
    plot_confusion_matrix(y_test, y_pred, classes=[str(i) for i in range(10)])

def test_plot_roc_curve():
    y_test = np.random.randint(0, 10, 100)
    y_score = np.random.rand(100, 10)
    plot_roc_curve(y_test, y_score, n_classes=10)

def test_plot_learning_curve():
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = np.random.rand(5, 3)
    valid_scores = np.random.rand(5, 3)
    plot_learning_curve(train_sizes, train_scores, valid_scores)

def test_plot_decision_boundary():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    plot_decision_boundary(model, X_test, y_test)

def test_plot_latent_space():
    X, y = make_classification(n_features=50, n_redundant=0, n_informative=2, n_samples=100,  # Ensure n_samples > 30
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    tsne = TSNE(n_components=2, random_state=42)
    X_test_tsne = tsne.fit_transform(X_test)
    plot_latent_space(X_test_tsne, y_test)

