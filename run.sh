#!/bin/bash

# Container name
CONTAINER_NAME="server"

# Stop and remove any existing container with the same name
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the container
echo "Starting container: $CONTAINER_NAME"
docker run -d \
    --name $CONTAINER_NAME \
    -p 5000:5000 \
    -v $(pwd)/persisted:/app/persisted \
    server:latest

echo "Container $CONTAINER_NAME is running on port 5000"

