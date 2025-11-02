# Multi-stage build: Stage 1 - Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/display/frontend

# Copy frontend package files
COPY display/frontend/package.json display/frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy frontend source and config files
COPY display/frontend/src ./src
COPY display/frontend/index.html ./
COPY display/frontend/tsconfig.json ./
COPY display/frontend/tsconfig.app.json ./
COPY display/frontend/tsconfig.node.json ./
COPY display/frontend/vite.config.ts ./
COPY display/frontend/public ./public

# Build frontend with Vite (skipping strict TypeScript checks)
RUN npx vite build


# Stage 2 - Runtime with Python/Flask
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY display/backend/requirements.txt ./display/backend/
RUN pip install --no-cache-dir -r display/backend/requirements.txt

# Copy backend code
COPY display/backend/backend.py ./display/backend/

# Copy SQLite database
COPY conversation_data.db ./display/backend/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/display/frontend/dist ./display/frontend/dist

# Set working directory to backend
WORKDIR /app/display/backend

# Expose ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()"

# Start Flask app
CMD ["python", "backend.py"]
