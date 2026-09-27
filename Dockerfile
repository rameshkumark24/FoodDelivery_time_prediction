FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1 is the OpenMP runtime that LightGBM and scikit-learn load.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the web app needs at runtime (no datasets, plots or training code).
COPY delivery/ delivery/
COPY templates/ templates/
COPY models/delivery_model.joblib models/
COPY app.py gunicorn.conf.py ./

RUN useradd --create-home --uid 10001 appuser
USER appuser

# Hosting platforms (Render, Railway, Heroku...) override PORT at runtime.
ENV PORT=5000
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '5000'), timeout=4)"

# Workers, bind address and timeouts come from gunicorn.conf.py.
CMD ["gunicorn", "app:app"]
