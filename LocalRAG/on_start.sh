#!/bin/bash

# Run this sript to start redis & qdrant locally

docker container run --detach --publish 6379:6379 redis/redis-stack:latest
docker container run --detach --publish 6333:6333 -v /tmp/local_qdrant:/qdrant/storage qdrant/qdrant:latest