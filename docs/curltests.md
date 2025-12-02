# Test commands for GPU Controller at 145.38.189.62
# Make sure to update your API key from api_keys.json

export GPU_ID=451226aa-5ca5-460a-a6d1-5718fcc10e3c

# 1. Health Check (no authentication needed)
curl http://145.38.189.62:8000/health | jq

# 2. Discover Available GPUs (requires authentication)
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     http://145.38.189.62:8000/gpu/discover | jq

# 3. Get GPU Statistics
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     http://145.38.189.62:8000/gpu/stats | jq

# 4. Check specific GPU status (replace GPU_ID with actual ID from discover)
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     http://145.38.189.62:8000/gpu/$GPU_ID/status | jq

# 5. Resume a GPU (replace GPU_ID with actual ID)
curl -X POST \
     -H "Authorization: Bearer $SURFLLM_API_KEY" \
     http://145.38.189.62:8000/gpu/$GPU_ID/resume

# 6. Pause a GPU (replace GPU_ID with actual ID)
curl -X POST \
     -H "Authorization: Bearer $SURFLLM_API_KEY" \
     http://145.38.189.62:8000/gpu/$GPU_ID/pause

# === OLLAMA PROXY TESTS (The magic happens here!) ===

# 7. Test Ollama Generate API (auto-scaling)
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     -H "Content-Type: application/json" \
     http://145.38.189.62:8000/api/generate -d '{
  "model": "phi3",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# 8. Test Ollama Generate API with streaming
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     -H "Content-Type: application/json" \
     http://145.38.189.62:8000/api/generate -d '{
  "model": "phi3", 
  "prompt": "Tell me a short story about a robot",
  "stream": true
}'

# 9. Test Ollama Chat API
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     -H "Content-Type: application/json" \
     http://145.38.189.62:8000/api/chat -d '{
  "model": "llama3:8b",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false
}'

# 10. Test OpenAI Compatible API
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     -H "Content-Type: application/json" \
     http://145.38.189.62:8000/v1/chat/completions -d '{
  "model": "llama3:8b",
  "messages": [
    {"role": "user", "content": "Tell me a joke"}
  ],
  "temperature": 0.7,
  "stream": false
}'

# 11. Test with different model (will auto-load)
curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
     -H "Content-Type: application/json" \
     http://145.38.189.62:8000/api/generate -d '{
  "model": "codellama:7b",
  "prompt": "Write a Python function to calculate fibonacci numbers",
  "stream": false
}'

# === STRESS TEST ===

# 12. Multiple concurrent requests (run in parallel)
for i in {1..5}; do
  curl -H "Authorization: Bearer $SURFLLM_API_KEY" \
       -H "Content-Type: application/json" \
       http://145.38.189.62:8000/api/generate -d '{
    "model": "phi3",
    "prompt": "Request '$i': What is machine learning?",
    "stream": false
  }' &
done
wait

echo "All tests completed!"
