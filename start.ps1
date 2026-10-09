$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
$taskImportCheck = @'
import sys
try:
    import fastapi, uvicorn, multipart, librosa, demucs, torch, torchaudio, onnxruntime, mido
except Exception:
    sys.exit(1)
'@
& $taskPython -c $taskImportCheck
if ($LASTEXITCODE -ne 0) {
    & $taskPython -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Please check your network.' }
    & $taskPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Please check your network.' }
}
Write-Host 'Open http://127.0.0.1:8765 in your browser. Keep this window open.'
& $taskPython -m uvicorn app.main:app --host 127.0.0.1 --port 8765
