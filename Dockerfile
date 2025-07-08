# Use the official Python image as a base
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-root --no-dev

# Copy the rest of your application code
COPY . .

# Expose the port for Ollama if you plan to run it in a separate container and link them
# This is not for Globule itself, but for services it connects to.
# ENV GLOBULE_LLM_BASE_URL="http://ollama:11434" # Example if Ollama is a linked service named 'ollama'
# ENV GLOBULE_EMBEDDING_BASE_URL="http://ollama:11434" # Example if Ollama is a linked service named 'ollama'

# Define the entry point for the container
# This makes the 'globule' command available
ENTRYPOINT ["poetry", "run", "globule"]
