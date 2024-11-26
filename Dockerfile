FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m celeryuser

# Create media directories with proper permissions
RUN mkdir -p /app/media/tmp /app/media/processed && \
    chown -R celeryuser:celeryuser /app/media && \
    chmod -R 775 /app/media

# Copy project files
COPY . .
RUN chown -R celeryuser:celeryuser /app

# Add entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh && \
    chown celeryuser:celeryuser /docker-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DJANGO_SETTINGS_MODULE=ankibyte.settings.development

USER celeryuser

ENTRYPOINT ["/docker-entrypoint.sh"]