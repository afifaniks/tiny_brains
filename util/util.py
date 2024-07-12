import torch


def print_gpu_memory_usage():
    print("Max memory allocated:", torch.cuda.max_memory_allocated() / (1024*1024), "MB")
    print("Current memory allocated:", torch.cuda.memory_allocated() / (1024*1024), "MB")
    print("Cache memory allocated:", torch.cuda.memory_reserved() / (1024*1024), "MB")