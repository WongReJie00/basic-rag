# OVMS
This folder contains ovms configuration to easily host models on ovms. There is a docker-compose.yml that spins up one container to download and extract the models to `ovms/workspace`, then spins up ovms to host the models on port 8000.

> [!NOTE]
> The models will not be redownloaded if they already exist. Also, remember to cd to the ovms directory.

# Usage
```bash
cd ovms
# export UID and GID variables (MUST SET for permissions to work correctly)
export UID=$(id -u)
export GID=$(id -g)
# pull models and start ovms
docker compose up -d
# stop ovms
docker compose down
# check if it works properly when ovms container is started
curl http://localhost:8000/v1/config
```

# Configuration
Change these settings in the docker-compose.yml. Use the huggingface name for the models.

For now, you can only use one embeddings and one reranker model, but you can put multiple models in text generation models, comma delimited.
```yaml
    environment:
      UID: $UID
      GID: $GID
      EMBEDDINGS_SOURCE_MODEL: "BAAI/bge-base-en-v1.5"
      RERANK_SOURCE_MODEL: "BAAI/bge-reranker-base"
      TEXT_GENERATION_SOURCE_MODELS: "microsoft/Phi-3-mini-128k-instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```