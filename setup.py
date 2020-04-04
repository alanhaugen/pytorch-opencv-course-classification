from setuptools import setup

setup(
        name='trainer',
        version='1.0',
        description='Train a kenyan food dataset with a classification model with PyTorch',
        author='Alan Haugen',
        author_email='alanhaugen@gmail.com',
        packages=['trainer'],  #same as name
        install_requires=['pandas', 'numpy', 'pytorch', 'torchvision', 'tensorboard', 'kaggle'], #external packages as dependencies
)
