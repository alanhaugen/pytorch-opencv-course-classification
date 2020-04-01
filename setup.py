from setuptools import setup

setup(
        name='source',
        version='1.0',
        description='This is made for a kaggle pytorch competition. I have never written a standard python project before',
        author='Alan Haugen',
        author_email='alanhaugen@gmail.com',
        packages=['source'],  #same as name
        install_requires=['numpy', 'pytorch', 'torchvision', 'tensorboard'], #external packages as dependencies
)
