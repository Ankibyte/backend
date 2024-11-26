#!/bin/bash

set -e

# Activate virtual environment
source /opt/venv/bin/activate

# Wait for database
echo "Waiting for database..."
python wait_for_db.py

# Only run migrations if this is the backend service
if [[ "$*" == *"runserver"* ]]; then
    echo "Running migrations..."
    python manage.py migrate --noinput
fi

echo "Starting application..."
exec "$@"