# Use the specified Python image from Microsoft Container Registry
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install Python packages
RUN pip3 --disable-pip-version-check --no-cache-dir install -r tensorflow pandas scikit-learn matplotlib pylint autopep8

WORKDIR /workspace