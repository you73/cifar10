# CIFAR-10 Classifier

A Python library for CIFAR-10 classification using traditional machine learning methods.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)
    - [Install Dependencies](#install-dependencies)
3. [Usage](#usage)
    - [Prepare Data](#prepare-data)
    - [Run Benchmark](#run-benchmark)
    - [Run Tests](#run-tests)
4. [Makefile Commands](#makefile-commands)
5. [Project Structure](#project-structure)
6. [Objective and Features](#objective-and-features)
7. [Example Usage](#example-usage)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)
11. [Contributing](#contributing)
12. [License](#license)
13. [Authors](#authors)
14. [Acknowledgments](#acknowledgments)
15. [References](#references)

## Introduction

This project aims to provide tools for benchmarking traditional machine learning algorithms on the CIFAR-10 dataset. The CIFAR-10 dataset is a well-known dataset in the field of machine learning, consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project includes functionalities for preprocessing images, extracting features, training models, evaluating models, and visualizing results.

## Installation

### Prerequisites

Ensure you have Python 3.6 or higher installed. You can download Python from the [official website](https://www.python.org/downloads/).

### Create and Activate Virtual Environment

To avoid conflicts with other projects, it is recommended to create a virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

### Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Usage

### Prepare Data

Prepare the CIFAR-10 dataset by downloading and extracting it:

```bash
make prepare-data
```

### Run Benchmark

Run benchmarks with specified preprocessor, feature extractor, and classifier:

```bash
make benchmark PREPROCESSOR=<preprocessor> FEATURE_EXTRACTOR=<feature_extractor> CLASSIFIER=<classifier> [BATCH_SIZE=<batch_size>] [LEARNING_RATE=<learning_rate>] [KERNEL=<kernel>]
```

Note: The options `BATCH_SIZE`, `LEARNING_RATE`, and `KERNEL` are optional and only used with the `sgd` classifier. For other classifiers, only the `PREPROCESSOR`, `FEATURE_EXTRACTOR`, and `CLASSIFIER` are required.

### Run Tests

Execute the test suite to ensure everything is functioning correctly:

```bash
make test
```

## Makefile Commands

To simplify the execution of common tasks, the following Makefile commands are available:

- **Install dependencies:** `make install`
- **Prepare data:** `make prepare-data`
- **Run benchmark:** `make benchmark PREPROCESSOR=<preprocessor> FEATURE_EXTRACTOR=<feature_extractor> CLASSIFIER=<classifier> [BATCH_SIZE=<batch_size>] [LEARNING_RATE=<learning_rate>] [KERNEL=<kernel>]`
- **Run tests:** `make test`
- **Clean up generated files:** `make clean`

## Project Structure

```plaintext
├── LICENSE
├── Makefile
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── setup.py
├── src
│   ├── benchmark.py
│   ├── data
│   │   └── dataset.py
│   ├── features
│   │   └── feature_extraction.py
│   ├── models
│   │   ├── model_evaluation.py
│   │   └── model_training.py
│   ├── preprocessing
│   │   └── data_preprocessing.py
│   ├── utils
│   │   └── io_utils.py
│   └── visualization
│       └── visualization_tools.py
├── tests
│   ├── conftest.py
│   ├── pytest.ini
│   ├── test_data_preprocessing.py
│   ├── test_dataset.py
│   ├── test_feature_extraction.py
│   ├── test_model_evaluation.py
│   ├── test_model_training.py
│   └── test_visualization.py
```

## Objective and Features

This library aims to provide tools for benchmarking traditional machine learning algorithms on the CIFAR-10 dataset. It includes functionalities for:

- **Preprocessing images:** Various techniques to preprocess images, such as normalization, ZCA whitening, and edge detection.
- **Extracting features:** Multiple methods to extract features from images, including Histogram of Oriented Gradients (HOG), Scale-Invariant Feature Transform (SIFT), and color histograms.
- **Training models:** Different machine learning models to train on the extracted features, such as Support Vector Machine (SVM), Stochastic Gradient Descent (SGD), and Random Forest.
- **Evaluating models:** Comprehensive evaluation metrics to assess model performance, including accuracy, precision, recall, F1 score, and ROC AUC.
- **Visualizing results:** Tools to visualize results and model performance, including confusion matrices, ROC curves, learning curves, and decision boundaries.

### Available Methods

#### Preprocessors
- `zca_whitening`
- `standardize`
- `normalize`
- `grayscale`
- `gaussian_filter`
- `median_filter`
- `binarize`
- `detect_edges`
- `erode`
- `dilate`

#### Feature Extractors
- `flatten`
- `hog`
- `color_histogram`
- `sift`
- `orb`
- `lbp`

#### Classifiers
- `sgd`
- `logistic_regression`
- `knn`
- `random_forest`
- `svm`
- `decision tree`



## Example Usage

Here's an example of how to run the benchmark with specific parameters:

```bash
make benchmark PREPROCESSOR=normalize FEATURE_EXTRACTOR=hog CLASSIFIER=knn
```

## Configuration

You can configure various aspects of the benchmark by modifying the parameters passed to the `make benchmark` command. Refer to the [benchmark script](src/benchmark.py) for detailed information on available options.

## Troubleshooting

If you encounter any issues, consider the following:

- **Ensure all dependencies are installed:** Run `pip install -r requirements.txt` and `pip install -r requirements-dev.txt` to ensure all dependencies are installed.
- **Activate the virtual environment:** Ensure the virtual environment is activated using `source myenv/bin/activate` (or `myenv\Scripts\activate` on Windows).
- **Check for typos:** Verify that the commands are typed correctly, especially the parameters for the `make benchmark` command.

## FAQ

**Q:** How do I update the dependencies?
**A:** You can update the dependencies by running `pip install --upgrade -r requirements.txt` and `pip install --upgrade -r requirements-dev.txt`.

**Q:** How do I deactivate the virtual environment?
**A:** Run `deactivate` to deactivate the virtual environment.

**Q:** Where are the models saved?
**A:** The trained models are saved in the `saved_models` directory with a name that includes the model, preprocessor, and feature extractor used.

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Open a pull request.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Authors

- Julien Cardi - [GitHub](https://github.com/devvv1337/)
- Youssef Agoulif - [GitHub](https://github.com/you73/)

For more details, visit the [GitHub repository](https://github.com/devvv1337/cifar10-lib).

## Acknowledgments

- The CIFAR-10 dataset creators.
- The contributors to the various open-source libraries used in this project.
- The open-source community for providing valuable tools and resources.

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [OpenCV Documentation](https://docs.opencv.org/master/)
- [Python Official Documentation](https://docs.python.org/3/)

---

Feel free to reach out if you have any questions or need further assistance. Happy coding!
```