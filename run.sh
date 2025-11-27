#!/bin/bash

# Container name
CONTAINER_NAME="guidance-app"

# Stop and remove any existing container with the same name
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Volume name for persisted data
VOLUME_NAME="guidance-app-data"

# Create volume if it doesn't exist
if [ ! "$(docker volume ls -q -f name=$VOLUME_NAME)" ]; then
    echo "Creating volume: $VOLUME_NAME"
    docker volume create $VOLUME_NAME
fi

# Run the container
echo "Starting container: $CONTAINER_NAME"
docker run -d \
    --name $CONTAINER_NAME \
    -p 127.0.0.1:5000:5000 \
    -v $VOLUME_NAME:/app/persisted \
    guidance-app:latest

echo "Container $CONTAINER_NAME is running on localhost:5000"
echo "Persisted data is stored in Docker volume: $VOLUME_NAME"

