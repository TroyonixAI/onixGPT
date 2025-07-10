# OnixGPT: Modular Financial GPT for Troyonix

OnixGPT is a lightweight, fine-tunable GPT framework designed for financial text modeling and agent workflows within the Troyonix AI wealth management platform. It is optimized for Apple Silicon (M1/M2) with torch-metal support, and is ready for local training as well as future FastAPI API serving.

## Features
- **Minimal, modular codebase** for easy customization and integration
- **Apple Silicon (M1/M2) compatible** with `torch-metal` GPU acceleration
- **Finance-focused**: Organize your financial data in `data/finance_notes/` (see below)
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
Organize your financial training data in the following structure:
```
data/finance_notes/
  raw/         # Place your original, unprocessed financial data here
  processed/   # Store preprocessed or cleaned data ready for training
  finetuned_models/ # Save your trained or fine-tuned model checkpoints here
```
- Use preprocessing scripts to convert raw data into the required binary format for training (see project scripts).

### 3. Train OnixGPT (Small Model Example)
```sh
python train.py config/train_onixgpt_finance.py --device=mps --compile=False
```
- For fine-tuning, use:
```sh
python train.py config/finetune_finance.py --device=mps --compile=False
```
- For character-level training, use:
```sh
python train.py config/train_finance_char.py --device=mps --compile=False
```
- Adjust model size, block size, and training steps in the respective config files in `config/`.
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
- `data/finance_notes/` – Folder for all financial training data (raw, processed, finetuned_models)
- `config/train_onixgpt_finance.py` – Training config (small, M1-friendly)
- `config/finetune_finance.py` – Fine-tuning config for finance
- `config/train_finance_char.py` – Character-level training config for finance
- `generate.py` – Script to generate text from a prompt
- `model.py` – Modular GPT model definition

## License
OnixGPT is released under the MIT License. See LICENSE for details.

---
**[Troyonix](https://troyonix.com/): Smarter AI for Wealth Management.**
