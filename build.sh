#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies with pip
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt 