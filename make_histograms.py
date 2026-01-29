import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

DATA_FILES = sys.argv[1:]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLOCK_SIZE = 10000

print(f'=== CREATING HISTOGRAMS FOR {len(DATA_FILES)} FILES ===')
for data_file in DATA_FILES:
    print(data_file)

for data_file in DATA_FILES:
    data = torch.load(data_file).to(DEVICE)
    print(data.shape)
    data_normalized = F.normalize(data, p=2, dim=1)

    similarity_scores = torch.empty((0,), device=DEVICE)

    for i in tqdm.tqdm(range(int((data_normalized.size(0) / BLOCK_SIZE) + 1))):
        x = data_normalized[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] @ data_normalized.T
        mask = torch.triu(torch.ones_like(x, dtype=torch.bool), diagonal=(i*BLOCK_SIZE)+1)
        similarity_scores = torch.cat((similarity_scores, x[mask]))

    histogram, boundaries = torch.histogram(similarity_scores.cpu(), bins=256, density=True)
    x_values = 0.5 * (boundaries[:-1] + boundaries[1:])

    np.savez(f'{data_file.split(".")[0]}.npz', x_values=x_values, histogram=histogram)

