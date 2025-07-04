# OnixGPT finance fine-tuning config (Apple Silicon friendly)
# Edit model size, block_size, and training steps for your hardware and data

out_dir = 'out-onixgpt-finance-ft'  # Output directory for fine-tuned checkpoints

# Evaluation and logging
# Increase eval_interval for larger datasets
# Decrease for quick feedback on small data

eval_interval = 250
log_interval = 10
always_save_checkpoint = False

eval_iters = 200

# Weights & Biases logging (optional)
wandb_log = False
wandb_project = 'onixgpt-finance'
wandb_run_name = 'ft-onixgpt-finance'

dataset = 'finance_notes'  # Looks for data/finance_notes.txt
init_from = 'gpt2'  # Start from base GPT-2 (or change to another checkpoint if desired)

# Training hyperparameters
batch_size = 32  # Lower if you run out of memory
block_size = 128  # Context window (tokens)

# Model size (small for MacBook Air)
n_layer = 4  # Number of transformer layers
n_head = 4   # Number of attention heads
n_embd = 128 # Embedding size
dropout = 0.1  # Increase for regularization on larger data

learning_rate = 1e-3
max_iters = 3000  # Increase for more training
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Apple Silicon/MacBook: set device and disable compile for MPS
# device = 'mps'  # Uncomment to force Apple GPU
# compile = False # Uncomment to disable torch.compile 