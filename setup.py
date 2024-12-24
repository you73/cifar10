from setuptools import find_packages, setup

setup(
    name='cifar10_classifier',
    version='0.1.0',
    description='A Python library for CIFAR-10 classification using traditional machine learning methods',
    author='Julien Cardi & Youssef Agoulif',
    author_email='265julien@gmail.com',
    url='https://github.com/devvv1337/cifar10-lib',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn',
        'scikit-image',
        'opencv-python-headless',
        'numpy',
        'matplotlib'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
