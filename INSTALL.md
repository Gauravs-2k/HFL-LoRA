# Installation Instructions

## Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 12.7

## Installation Steps

### 1. Create Virtual Environment (if not already created)
```powershell
python -m venv dml_proj
.\dml_proj\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA Support
**IMPORTANT**: Install PyTorch FIRST before other dependencies:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Other Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify CUDA Installation
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

You should see:
```
CUDA available: True
CUDA version: 12.4
```

## Running the Script

**Always run from the project root directory:**
```powershell
cd C:\Users\garvi\OneDrive\Desktop\DML
python -m app.federation.department_client
```

Or use the provided batch file:
```powershell
.\run_client.bat
```

## Using GPU Acceleration
To use GPU (recommended for much faster training):
```powershell
python -m app.federation.department_client --device-map auto
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'app'
**Solution**: Make sure you're running from the DML root directory, not from subdirectories.

### Issue: CUDA not available
**Solution**: Reinstall PyTorch with CUDA support (see step 2 above).
