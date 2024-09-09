# Use an official Python runtime as a parent image
FROM python:3.10.14-slim

# Set environment variables to prevent Python from writing pyc files and to keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies for Python
# RUN apt-get update && apt-get install -y gcc

# Copy only the pyproject.toml and other necessary files for installing dependencies
# COPY pyproject.toml .


# Copy the entire application code into the container
COPY . .

# Install dependencies (excluding dev dependencies)
RUN pip install .[serve,eval]


# Run calibration during the build process
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.1 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.2 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.3 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.4 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.6 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.7 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.8 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.9 --config config.example.yaml
RUN python -m routellm.calibrate_threshold --routers mf --strong-model-pct 1 --config config.example.yaml


# Expose the port the app runs on
EXPOSE 6060

# Set environment variables for production mode (if needed)
ENV FLASK_ENV=production

# Set the default command to run your server
CMD ["python", "-m", "routellm.openai_server", "--routers", "mf", "--verbose"]
