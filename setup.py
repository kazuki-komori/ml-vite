from setuptools import setup, find_packages
from os import path

def get_version():
    version_filepath = path.join(path.dirname(__file__), 'ml_vite', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]

install_requires = [
  "pandas",
  "numpy",
  "tqdm"
]

setup(
  name="ml_vite",
  version=get_version(),
  install_requires=install_requires,
  author="kazuyan",
  packages=find_packages()
)