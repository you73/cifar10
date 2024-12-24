import numpy as np
import os
import requests
import tarfile

def download_and_extract_cifar10(download_dir="data/raw", extract_dir="data/processed"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(download_dir, "cifar-10-python.tar.gz")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10 dataset...")
        response = requests.get(url, stream=True)
        with open(tar_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

def load_cifar10(data_dir="data/processed/cifar-10-batches-py"):
    def load_batch(fpath):
        import pickle
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            data = d[b'data']
            labels = d[b'labels']
            return data, labels

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 32, 32, 3), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(data_dir, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_batch(fpath)
    x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test

def save_processed_data(x_train, x_test, y_train, y_test, processed_data_path="data/processed/"):
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    np.save(os.path.join(processed_data_path, "x_train.npy"), x_train)
    np.save(os.path.join(processed_data_path, "x_test.npy"), x_test)
    np.save(os.path.join(processed_data_path, "y_train.npy"), y_train)
    np.save(os.path.join(processed_data_path, "y_test.npy"), y_test)

def analyze_cifar10(x_train, y_train):
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Image size: {x_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")

if __name__ == "__main__":
    download_and_extract_cifar10()
    x_train, x_test, y_train, y_test = load_cifar10()
    save_processed_data(x_train, x_test, y_train, y_test)
    analyze_cifar10(x_train, y_train)
