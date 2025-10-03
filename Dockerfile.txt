# ============================================
# Base Image
# ============================================
FROM python:3.9-slim

# ============================================
# Working Directory
# ============================================
WORKDIR /app

# ============================================
# Install System Dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Copy Requirements and Install Dependencies
# ============================================
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ============================================
# Copy Application Code
# ============================================
COPY . .

RUN mkdir -p models templates static visualizations

# ============================================
# Environment Variables
# ============================================
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# ============================================
# Expose Port
# ============================================
EXPOSE 5000

# ============================================
# Health Check
# ============================================
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# ============================================
# Start the Application
# ============================================
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
