from setuptools import setup, find_packages

setup(
    name='sentimentanalyse-test-package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'pydantic',
        'strictyaml',
        'joblib',
        'pytest',
        'mypy',
        'tox',
        'black',
        'flake8',
        'isort',
        'nltk',
        'tqdm'
    ],
)