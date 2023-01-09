#!/bin/bash
cd "$(dirname "$0")"
cd ..

rm -rf build dist

python setup.py sdist bdist_wheel

twine upload --config-file .pypirc --repository testpypi dist/*
