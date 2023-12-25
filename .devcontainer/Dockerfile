# Use the official Python 3.11 image from Docker Hub.
FROM python:3.11-bullseye

# Set the working directory in the container to `/workspace`.
WORKDIR /workspace

# Copy the requirements file from your host to the current directory in the container.
COPY requirements.txt .

# Install any needed packages specified in `requirements.txt`.
RUN pip install --no-cache-dir -r requirements.txt