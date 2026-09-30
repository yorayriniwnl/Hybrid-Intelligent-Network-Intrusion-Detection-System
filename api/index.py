"""
Vercel Serverless Function Entry Point for Hybrid Intelligent NIDS
Exposes FastAPI app for serverless execution on Vercel
"""
import os
import sys

# Ensure project root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app import app
