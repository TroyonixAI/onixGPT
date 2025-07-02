# OnixGPT: Modular Financial LLM for Troyonix

OnixGPT is a lightweight, fine-tuneable GPT framework designed for financial text modeling and agent workflows in the Troyonix AI wealth management platform. Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), it is optimized for Apple Silicon (M1/M2) with torch-metal support, and is ready for local training and future FastAPI API serving.

## Features
- **Minimal, modular codebase** for easy customization and integration
- **Apple Silicon (M1/M2) compatible** with `torch-metal` GPU acceleration
- **Finance-focused**: Start with your own `data/finance_notes.txt` or any financial corpus
- **Configurable model sizes** for MacBook Air and similar hardware
- **Ready for FastAPI**: Modular scripts and clear comments for API/agent integration

## Quick Start

### 1. Install Requirements
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
- For Apple Silicon GPU support, see [PyTorch Metal docs](https://pytorch.org/docs/stable/notes/mps.html).

### 2. Prepare Data
Place your financial training data in `data/finance_notes.txt` (plain text, one note per line).

### 3. Train OnixGPT (Small Model Example)
```sh
python train.py config/train_onixgpt_finance.py --device=mps --compile=False
```
- Adjust model size, block size, and training steps in `config/train_onixgpt_finance.py`.
- For CPU-only: use `--device=cpu`.

### 4. Generate Financial Text
```sh
python generate.py --out_dir=out-onixgpt-finance --start="Market update: " --device=mps
```
- See `generate.py` for prompt and output options.

## Modularity & Integration
- All scripts are **commented** for easy editing (model size, token limits, etc.)
- Designed for future FastAPI backend integration (see `generate.py` for modular function)
- Add new configs in `config/` for different domains or model sizes

## Directory Structure
- `data/finance_notes.txt` – Your financial training data
- `config/train_onixgpt_finance.py` – Training config (small, M1-friendly)
- `generate.py` – Script to generate text from a prompt
- `model.py` – Modular GPT model definition

## License
OnixGPT is based on nanoGPT (MIT License). See LICENSE for details.

---
**Troyonix: Smarter AI for Wealth Management.**
