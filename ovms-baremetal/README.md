# OVMS Baremetal Setup

This directory contains instructions and files for running OpenVINO Model Server (OVMS) directly on your machine (baremetal), without Docker.

## Prerequisites

- Linux OS
- Python 3.8+
- [OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_docs_deploying_server_baremetal.html#) installed
- Required models downloaded

## Installation

Follow the official guide to install OVMS:
https://docs.openvino.ai/2025/model-server/ovms_docs_deploying_server_baremetal.html#

## Usage

1. Prepare your model directory (see `models/` example).
2. Start OVMS using a command like:

```bash
ovms --config_path /path/to/config.json --port 8000
```

3. Update the backend and Streamlit app to connect to your baremetal OVMS endpoint (default: `http://localhost:8000/v3`).

## Files

- `baremetal_backend.py`: Backend for connecting to OVMS running on baremetal
- `baremetal_app.py`: Streamlit UI for interacting with the baremetal OVMS backend

## Example Model Directory Structure

```
models/
  BAAI/
    bge-base-en-v1.5/
      ...
  Qwen/
    Qwen2-7B-Instruct/
      ...
```

## Example OVMS Start Command

```bash
ovms --config_path ./config.json --port 8000
```

## Notes

- Make sure the model paths in your config match your local directory structure.
- You can run multiple OVMS instances for different models if needed.
