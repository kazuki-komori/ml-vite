from setuptools import setup, find_packages

install_requires = [
  "pandas",
  "numpy",
  "tqdm"
]

setup(
  name="ml_vite",
  version="0.0.1",
  install_requires=install_requires,
  author="kazuyan",
  packages=find_packages()
)