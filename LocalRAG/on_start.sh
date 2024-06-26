#!/bin/bash

#Run this on linux machine to run the ollama
mkdir bin
sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /bin/ollama
sudo chmod +x /bin/ollama
./bin/ollama pull llama3
mkdir -p ~/log
~/bin/ollama serve > ~/log/ollama.log 2> ~/log/ollama.err &

# Run this to start redis & qdrant locally
docker container run --detach --publish 6379:6379 redis/redis-stack:latest
docker container run --detach --publish 6333:6333 -v /tmp/local_qdrant:/qdrant/storage qdrant/qdrant:latest