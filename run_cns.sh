#!/bin/bash

IMAGE_NAME="humungous-cns"
CONTAINER_NAME="humungous-cns"
DOCKERFILE_NAME="Dockerfile"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "🔧 Image '$IMAGE_NAME' not found — building it now..."
    docker build -t "$IMAGE_NAME" -f $DOCKERFILE_NAME .
else
    echo "✅ Image '$IMAGE_NAME' already exists — skipping build."
fi

# Stop and remove any existing container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🧹 Removing existing container named $CONTAINER_NAME..."
    docker rm -f "$CONTAINER_NAME"
fi

# Run the container
docker run --gpus all -it  --runtime=nvidia  -v "$(pwd):/workspace" --workdir /workspace -p 8000:8000 "$IMAGE_NAME" bash -c "apt update && apt install -y dos2unix && dos2unix src/CNS_venv/bin/activate && source src/CNS_venv/bin/activate && bash"

# Connect to the container
echo "🔗 Connecting to container..."
# echo "  yay "
docker exec -it "$CONTAINER_NAME" bash