# Use the specified Python image from Microsoft Container Registry
FROM mcr.microsoft.com/devcontainers/python:0-3.10-bullseye

# Install Python packages
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

WORKDIR /workspace