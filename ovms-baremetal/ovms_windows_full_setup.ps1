# OVMS Windows setup: download, convert, and serve TinyLlama
# Run in PowerShell

# --------- PYTHON ENV ---------
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# --------- DOWNLOAD REQUIREMENTS & EXPORT SCRIPT ---------
if (!(Test-Path "workspace")) { mkdir workspace }
cd workspace
Invoke-WebRequest https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/export_models/requirements.txt -OutFile requirements.txt
pip install -r requirements.txt
Invoke-WebRequest https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/export_models/export_model.py -OutFile export_model.py
cd ..

# --------- DOWNLOAD & CONVERT MODELS ---------
if (!(Test-Path "models")) { mkdir models }
$models = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int8-ov".Split(",")
foreach ($model in $models) {
    $modelPath = "models/" + $model.Replace("/", "\\")
    if (!(Test-Path $modelPath)) {
        python workspace/export_model.py text_generation --source_model $model --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --target_device GPU --cache_size 3
    }
    Write-Host "Model $model prepared."
}

# --------- DOWNLOAD & EXTRACT OVMS ---------
if (!(Test-Path ".\ovms\ovms.exe")) {
    Invoke-WebRequest https://github.com/openvinotoolkit/model_server/releases/download/v2025.2.1/ovms_windows_python_on.zip -OutFile ovms.zip
    Expand-Archive ovms.zip -DestinationPath . -Force
}

# --------- SETUP ENV ---------
.\ovms\setupvars.ps1

# --------- SERVE MODELS ---------
Write-Host "Starting OVMS on port 8002 with config models/config_all.json..."
.\ovms\ovms.exe --rest_port 8002 --config_path .\models\config_all.json
