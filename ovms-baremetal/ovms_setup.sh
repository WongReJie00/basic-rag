#!/bin/bash
# OVMS download, extract, and serve script
set -e
export HTTP_PROXY="http://proxy-dmz.intel.com:912"
export HTTPS_PROXY="http://proxy-dmz.intel.com:912"
export http_proxy="http://proxy-dmz.intel.com:912"
export https_proxy="http://proxy-dmz.intel.com:912"
# --------- CONFIGURATION ---------
PORT=8002
OVMS_VERSION="v2025.2.1"
OVMS_PACKAGE="ovms_ubuntu24.tar.gz"
OVMS_URL="https://github.com/openvinotoolkit/model_server/releases/download/${OVMS_VERSION}/${OVMS_PACKAGE}"
OVMS_DIR="$(pwd)/ovms"
OVMS_BIN_DIR="$OVMS_DIR/bin"
OVMS_LIB_DIR="$OVMS_DIR/lib"
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
