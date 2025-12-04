#!/bin/bash

# Start Ollama in the background.
ollama serve &
pid=$!

# Wait for Ollama to start.
echo "Waiting for Ollama to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama started!"

# Pull models specified in OLLAMA_MODELS environment variable.
if [ -n "$OLLAMA_MODELS" ]; then
    IFS=',' read -ra MODELS <<< "$OLLAMA_MODELS"
    for model in "${MODELS[@]}"; do
        echo "Pulling model: $model"
        ollama pull "$model"
    done
fi

echo "All models pulled. Ready to serve."

# Wait for the Ollama process to finish.
wait $pid
