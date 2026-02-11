# 🔧 Environment Setup — `finetune-sam`

This project uses a Conda environment named **`finetune-sam`** with:

- **Python 3.9.23**
- **PyTorch 2.1.0**
- **CUDA 12 runtime wheels**
- Open3D, OpenCV, MMDetection stack, Ultralytics, FastAPI, Gradio, etc.

---

## 🚀 Create Environment

```bash
conda create -n finetune-sam python=3.9.23 -y
conda activate finetune-sam
sudo apt update
sudo apt install -y build-essential git libgl1 libglib2.0-0
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0
pip install -r requirements.txt

---

## 🔥 Run Editor

```bash
conda activate finetune-sam
cd *PROJECT*/MCAF/backend
uvicorn main:app --reload --port 8000
