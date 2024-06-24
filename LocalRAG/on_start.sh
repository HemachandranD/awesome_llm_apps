#!/bin/bash

# Ru this sript to start Ollama locally

mkdir -p ~/log
~/bin/ollama serve > ~/log/ollama.log 2> ~/log/ollama.err &
docker container run --detach --publish 6379:6379 redis/redis-stack:latest
docker container run --detach --publish 6333:6333 -v /tmp/local_qdrant:/qdrant/storage qdrant/qdrant:latest