"""
Generate text from a trained OnixGPT model.
Modular, Apple Silicon (M1/M2) and torch-metal compatible.
Ready for FastAPI integration (see generate_text function).
"""
import os
import pickle
import argparse
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

def generate_text(prompt, out_dir, device='cpu', max_new_tokens=200, temperature=0.8, top_k=100, compile_model=False):
    """
    Generate text from a trained OnixGPT model.
    Args:
        prompt (str): The input prompt string.
        out_dir (str): Directory containing the trained model checkpoint.
        device (str): 'cpu', 'cuda', or 'mps' (Apple Silicon GPU).
        max_new_tokens (int): Number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling.
        compile_model (bool): Use torch.compile for speed (not supported on MPS).
    Returns:
        str: The generated text.
    """
    # Set up device and dtype
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile_model and device_type == 'cuda':
        model = torch.compile(model)

    # Load encoding
    meta_path = os.path.join('data', checkpoint.get('config', {}).get('dataset', ''), 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # Encode prompt
    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # Generate
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            return decode(y[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained OnixGPT model.")
    parser.add_argument('--prompt', type=str, default="Market update: ", help='Prompt to start generation')
    parser.add_argument('--out_dir', type=str, default='out-onixgpt-finance', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cpu', help="Device: 'cpu', 'cuda', or 'mps'")
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=100, help='Top-k sampling')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile (CUDA only)')
    args = parser.parse_args()

    # Print device info
    print(f"Using device: {args.device}")
    print(f"Generating with prompt: {args.prompt}")
    print("---")
    output = generate_text(
        prompt=args.prompt,
        out_dir=args.out_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        compile_model=args.compile
    )
    print(output) 