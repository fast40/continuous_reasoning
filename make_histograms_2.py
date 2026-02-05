import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')


# print(model.get_input_embeddings())

DATA_FILES = sys.argv[1:]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLOCK_SIZE = 2048

data = model.get_input_embeddings().weight.detach()
print(data.requires_grad)
random_indices = torch.randperm(data.size(0) // 10)
data = data[random_indices]
print(data.requires_grad)

# data = model.model.layers[0].input_layernorm(data).detach()
# print(data.requires_grad)

print(data.shape)
data_normalized = F.normalize(data, p=2, dim=1)

similarity_scores = torch.empty((0,))

for i in tqdm.tqdm(range(int((data_normalized.size(0) / BLOCK_SIZE) + 1))):
    x = data_normalized[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] @ data_normalized.T
    mask = torch.triu(torch.ones_like(x, dtype=torch.bool), diagonal=(i*BLOCK_SIZE)+1)
    similarity_scores = torch.cat((similarity_scores, x[mask].cpu()))

histogram, boundaries = torch.histogram(similarity_scores.cpu(), bins=256, density=True)
x_values = 0.5 * (boundaries[:-1] + boundaries[1:])

# np.savez(f'{data_file.split(".")[0]}.npz', x_values=x_values, histogram=histogram)
np.savez(f'hist_prenorm.npz', x_values=x_values, histogram=histogram)

