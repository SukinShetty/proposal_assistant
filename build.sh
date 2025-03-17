#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Running custom build script"
echo "Python version:"
python --version
echo "Pip version:"
pip --version

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing wheel and setuptools..."
pip install -U wheel setuptools

echo "Installing dependencies..."
pip install -r requirements.txt 