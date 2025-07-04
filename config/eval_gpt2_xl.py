# Evaluate the base gpt2-xl model with OnixGPT
# n_layer=48, n_head=25, n_embd=1600
# 1.5B parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'gpt2-xl'
