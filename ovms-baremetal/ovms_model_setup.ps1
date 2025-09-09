
# --------- USER CONFIGURABLE MODELS ---------
$EMBEDDINGS_SOURCE_MODEL = "BAAI/bge-base-en-v1.5"
$RERANK_SOURCE_MODEL = "BAAI/bge-reranker-base"

# --------- WORKSPACE SETUP ---------
if (!(Test-Path "workspace")) { mkdir workspace }
Set-Location workspace

# --------- PYTHON ENV ---------
if (!(Test-Path ".venv")) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1

# --------- DOWNLOAD EXPORT SCRIPT & INSTALL REQUIREMENTS ---------
if (!(Test-Path "export_model.py")) {
    Invoke-WebRequest https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/export_model.py -OutFile export_model.py
    pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt
}


# --------- CREATE MODELS DIR IF NEEDED ---------
if (!(Test-Path "models")) { mkdir models }

# --------- EXPORT EMBEDDINGS MODEL ---------
if (!(Test-Path "models/$EMBEDDINGS_SOURCE_MODEL")) {
    python export_model.py embeddings_ov --source_model $EMBEDDINGS_SOURCE_MODEL --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
}

# --------- EXPORT RERANKER MODEL ---------
if (!(Test-Path "models/$RERANK_SOURCE_MODEL")) {
    python export_model.py rerank_ov --source_model $RERANK_SOURCE_MODEL --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
}

# --------- RETURN TO BASE DIR ---------
Set-Location ..

# --------- DOWNLOAD & EXTRACT OVMS ---------
if (!(Test-Path ".\ovms\ovms.exe")) {
    Invoke-WebRequest https://github.com/openvinotoolkit/model_server/releases/download/v2025.2.1/ovms_windows_python_on.zip -OutFile ovms.zip
    Expand-Archive ovms.zip -DestinationPath . -Force
}

# --------- SETUP OVMS ENV ---------
.\ovms\setupvars.ps1

# --------- SERVE MODELS ---------
Write-Host "Starting OVMS on port 8002 with config workspace/models/config.json..."
.\ovms\ovms.exe --rest_port 8002 --config_path .\workspace\models\config.json
