.PHONY: docker-build docker-run docker-send docker-stop help

# Docker configuration
IMAGE_NAME := ausdevs-viz
IMAGE_TAG := latest
DOCKER_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)
CONTAINER_NAME := ausdevs
REMOTE_SERVER := root@150.107.75.167
REMOTE_PATH := /opt/ausdevs

help:
	@echo "AusDevs 2.0.0 - Docker Commands"
	@echo "================================="
	@echo ""
	@echo "Docker targets:"
	@echo "  make docker-build     - Build the Docker image locally"
	@echo "  make docker-run       - Run the container locally (port 5000)"
	@echo "  make docker-stop      - Stop the running container"
	@echo "  make docker-send      - Build, save, and send image to remote server"
	@echo ""

# Build Docker image
docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "✓ Image built successfully"

# Run Docker container locally
docker-run: docker-build
	@echo "Starting container on port 5000..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		--restart unless-stopped \
		-p 5000:5000 \
		$(DOCKER_IMAGE)
	@echo "✓ Container running at http://localhost:5000"
	@echo "  To view logs: docker logs -f $(CONTAINER_NAME)"
	@echo "  To stop: make docker-stop"

# Stop running container
docker-stop:
	@echo "Stopping container..."
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "✓ Container stopped"

# Build and send to remote server
docker-send: docker-build
	@echo "Preparing to send image to remote server..."
	@echo "Server: $(REMOTE_SERVER)"
	@echo ""
	@echo "Saving Docker image..."
	docker save $(DOCKER_IMAGE) | gzip > $(IMAGE_NAME).tar.gz
	@echo "✓ Image saved ($(shell du -h $(IMAGE_NAME).tar.gz | cut -f1))"
	@echo ""
	@echo "Transferring to remote server..."
	scp $(IMAGE_NAME).tar.gz $(REMOTE_SERVER):$(REMOTE_PATH)/
	@echo "✓ Image transferred"
	@echo ""
	@echo "Deploying on remote server..."
	ssh $(REMOTE_SERVER) "cd $(REMOTE_PATH) && \
		echo 'Loading image...' && \
		docker load < $(IMAGE_NAME).tar.gz && \
		echo 'Stopping old container...' && \
		docker stop $(CONTAINER_NAME) 2>/dev/null || true && \
		docker rm $(CONTAINER_NAME) 2>/dev/null || true && \
		echo 'Starting new container...' && \
		docker run -d \
			--name $(CONTAINER_NAME) \
			--restart unless-stopped \
			-p 5000:5000 \
			$(DOCKER_IMAGE) && \
		echo '✓ Deployment complete' && \
		echo 'Cleaning up...' && \
		rm $(IMAGE_NAME).tar.gz && \
		docker image prune -f"
	@echo "✓ Remote deployment successful!"
	@echo ""
	@echo "Access the app at: http://150.107.75.167:5000"
	@echo ""
	@echo "Cleaning up local tar file..."
	rm $(IMAGE_NAME).tar.gz
	@echo "✓ Done"
