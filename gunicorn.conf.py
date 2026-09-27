"""Gunicorn settings, picked up automatically when gunicorn starts from the
project directory (``gunicorn app:app``). Every value can be overridden with
an environment variable, e.g. ``PORT`` (set by Render/Heroku/Railway)."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
# Load the model once in the master and share it with the workers
# (copy-on-write), saving ~100 MB of RAM. Safe because app.py pins
# OMP_NUM_THREADS=1; see the comment there.
preload_app = True
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "60"))
accesslog = "-"
errorlog = "-"
