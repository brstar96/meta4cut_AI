import numpy as np
import pandas as pd
# import tensorflow as tf
import torch

print(f'np: {np.__version__}')
print(f'pd: {pd.__version__}')
# print(f'tf: {tf.__version__}')
print(f'torch: {torch.__version__}')

print(f'cuda enable: {torch.cuda.is_available()}')
print(f'current_device: {torch.cuda.current_device()}')
print(f'device: {torch.cuda.device(0)}')
print(f'device_count: {torch.cuda.device_count()}')
print(f'get_device_name: {torch.cuda.get_device_name(0)}')