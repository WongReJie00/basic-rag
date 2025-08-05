#!/bin/bash
# Full OVMS baremetal setup: download, convert, and serve models
set -e
export HTTP_PROXY="http://proxy-dmz.intel.com:912"
export HTTPS_PROXY="http://proxy-dmz.intel.com:912"
export http_proxy="http://proxy-dmz.intel.com:912"
export https_proxy="http://proxy-dmz.intel.com:912"
# --------- CONFIGURATION ---------
EMBEDDINGS_SOURCE_MODEL="BAAI/bge-base-en-v1.5"
RERANK_SOURCE_MODEL="BAAI/bge-reranker-base"
TEXT_GENERATION_SOURCE_MODELS="Qwen/Qwen2-7B-Instruct"
PORT=8002
OVMS_VERSION="v2025.2.1"
OVMS_PACKAGE="ovms_ubuntu24.tar.gz"
OVMS_URL="https://github.com/openvinotoolkit/model_server/releases/download/${OVMS_VERSION}/${OVMS_PACKAGE}"
OVMS_DIR="$(pwd)/ovms"
OVMS_BIN_DIR="$OVMS_DIR/bin"
OVMS_LIB_DIR="$OVMS_DIR/lib"

# --------- PYTHON ENV ---------
sudo apt update
sudo apt install -y python3 python3-venv python3-pip wget git libxml2 curl
python3 -m venv .venv --prompt ovms-llm
source .venv/bin/activate

# --------- DOWNLOAD REQUIREMENTS & EXPORT MODEL SCRIPT ---------
mkdir -p workspace
cd workspace
wget -nc https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/export_models/requirements.txt 
pip install -r requirements.txt
wget -nc https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/export_models/export_model.py
cd ..

# --------- DOWNLOAD & EXPORT MODELS ---------
mkdir -p models
if [ ! -d models/$EMBEDDINGS_SOURCE_MODEL ]; then
  python workspace/export_model.py embeddings --source_model $EMBEDDINGS_SOURCE_MODEL --weight-format int8 --config_file_path models/config_rag.json
fi
if [ ! -d models/$RERANK_SOURCE_MODEL ]; then
  python workspace/export_model.py rerank --source_model $RERANK_SOURCE_MODEL --weight-format int8 --config_file_path models/config_rag.json
fi
IFS=',' read -ra MODELS <<< "$TEXT_GENERATION_SOURCE_MODELS"
for MODEL in "${MODELS[@]}"; do
  if [ ! -d models/$MODEL ]; then
    python workspace/export_model.py text_generation --source_model $MODEL --weight-format int8 --kv_cache_precision u8 --config_file_path models/config_rag.json --model_repository_path models
  fi
  chmod -R 755 models/$MODEL
  echo "Model $MODEL prepared."
done

# --------- DOWNLOAD & EXTRACT OVMS ---------
if [ ! -f "$OVMS_BIN_DIR/ovms" ]; then
  wget "$OVMS_URL" -O "$OVMS_PACKAGE"
  tar -xzvf "$OVMS_PACKAGE"
fi
export LD_LIBRARY_PATH="$OVMS_LIB_DIR"
export PATH="$PATH:$OVMS_BIN_DIR"

# --------- SERVE MODELS ---------
echo "Starting OVMS on port $PORT with config models/config_rag.json..."
"$OVMS_BIN_DIR/ovms" --rest_port $PORT --config_path $(pwd)/models/config_rag.json
