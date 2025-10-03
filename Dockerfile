FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for scientific libs and curl
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel for reliable builds
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file and install dependencies without cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Print numpy and pandas versions for build verification
RUN python -c "import numpy; print('Docker numpy version:', numpy.__version__)"
RUN python -c "import pandas; print('Docker pandas version:', pandas.__version__)"

# Copy app source code
COPY . .

# Create needed directories if missing
RUN mkdir -p models templates visualizations

# Expose container port 5000 (used by Flask/Gunicorn)
EXPOSE 5000

# Set environment variables for Flask and Python buffering
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Health check endpoint to confirm app readiness
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Start Gunicorn, listening on all interfaces and the port Render specifies
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:$PORT", "--timeout", "120", "app:app"]
