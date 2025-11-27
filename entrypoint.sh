#!/bin/bash
set -e

# Ensure persisted directories exist with proper permissions
mkdir -p /app/persisted/uploads/ASSI-A
mkdir -p /app/persisted/uploads/ASSI-C
mkdir -p /app/persisted/student_data/ASSI-A
mkdir -p /app/persisted/student_data/ASSI-C
mkdir -p /app/persisted/results/ASSI-A
mkdir -p /app/persisted/results/ASSI-C

# Set permissions (755 for directories)
chmod -R 755 /app/persisted 2>/dev/null || true

# Execute the main command
exec "$@"

