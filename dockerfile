# Use official Python image as base
FROM python:3.13.7

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port (change if your app uses a different port)
EXPOSE 5000

# Create volume for persisted data
VOLUME ["/app/persisted"]

# Set environment variables (optional)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Run the Flask app
CMD ["python", "app.py"]