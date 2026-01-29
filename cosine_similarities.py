import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

COSINE_SIMILARITIES_FILE = sys.argv[1] if len(sys.argv) > 1 else 'histogram.npz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLOCK_SIZE = 10000

model = AutoModelForCausalLM.from_pretrained('gpt2')

embeddings = model.transformer.wte.weight.detach().to(DEVICE)
# embeddings = torch.rand_like(embeddings) - 0.5
embeddings_normalized = F.normalize(embeddings, p=2, dim=1)

similarity_scores = torch.empty((0,), device=DEVICE)

print(f'Calculating all pairwise cosine similarities...', end=' ', flush=True)
t = time.perf_counter()

for i in range(int((embeddings.size(0) / BLOCK_SIZE) + 1)):
    x = embeddings_normalized[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] @ embeddings_normalized.T
    mask = torch.triu(torch.ones_like(x, dtype=torch.bool), diagonal=(i*BLOCK_SIZE)+1)
    similarity_scores = torch.cat((similarity_scores, x[mask]))

t = time.perf_counter() - t
print(f'Calculated {similarity_scores.size(0)} cosine similarities in {t} seconds.')

print(f'Generating histogram...', end=' ', flush=True)
t = time.perf_counter()

histogram, boundaries = torch.histogram(similarity_scores.cpu(), bins=1024, density=True)
x_values = 0.5 * (boundaries[:-1] + boundaries[1:])

t = time.perf_counter() - t
print(f'Generated histogram in {t} seconds.')

print(f'Saving histogram...', end=' ', flush=True)
t = time.perf_counter()

np.savez(COSINE_SIMILARITIES_FILE, x_values=x_values, histogram=histogram)

t = time.perf_counter() - t
print(f'Saved to {COSINE_SIMILARITIES_FILE} in {t} seconds.')

