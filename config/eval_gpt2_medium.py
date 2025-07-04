# Evaluate the base gpt2-medium model with OnixGPT
# n_layer=24, n_head=16, n_embd=1024
# 345M parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'gpt2-medium'
