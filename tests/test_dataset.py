import pytest
from data.dataset import download_and_extract_cifar10, load_cifar10

def test_download_and_extract_cifar10():
    download_and_extract_cifar10()

def test_load_cifar10():
    x_train, x_test, y_train, y_test = load_cifar10()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000,)
    assert y_test.shape == (10000,)
