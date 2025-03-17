from flask import Flask, request, jsonify, render_template, redirect, url_for
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app from the parent directory
from app import app as flask_app

# This is needed for Vercel
app = flask_app 