import time
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

DATA_FILES = sys.argv[1:]
# HISTOGRAM_FILE = f'{DATA_FILE.split(".")[0]}.npz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLOCK_SIZE = 10000

print(f'=== CREATING HISTOGRAMS FOR {len(DATA_FILES)} FILES ===')
for DATA_FILE in DATA_FILES:
    print(DATA_FILE)

for DATA_FILE in tqdm.tqdm(DATA_FILES):
    data = torch.load(DATA_FILE).to(DEVICE)
    data_normalized = F.normalize(data, p=2, dim=1)

    similarity_scores = torch.empty((0,), device=DEVICE)

    # print(f'Calculating all pairwise cosine similarities...', end=' ', flush=True)
    # t = time.perf_counter()

    for i in range(int((data_normalized.size(0) / BLOCK_SIZE) + 1)):
        x = data_normalized[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] @ data_normalized.T
        mask = torch.triu(torch.ones_like(x, dtype=torch.bool), diagonal=(i*BLOCK_SIZE)+1)
        similarity_scores = torch.cat((similarity_scores, x[mask]))

    # t = time.perf_counter() - t
    # print(f'Calculated {similarity_scores.size(0)} cosine similarities in {t} seconds.')
    #
    # print(f'Generating histogram...', end=' ', flush=True)
    # t = time.perf_counter()

    histogram, boundaries = torch.histogram(similarity_scores.cpu(), bins=1024, density=True)
    x_values = 0.5 * (boundaries[:-1] + boundaries[1:])

    # t = time.perf_counter() - t
    # print(f'Generated histogram in {t} seconds.')
    #
    # print(f'Saving histogram...', end=' ', flush=True)
    # t = time.perf_counter()

    np.savez(f'{DATA_FILE.split(".")[0]}.npz', x_values=x_values, histogram=histogram)

    # t = time.perf_counter() - t
    # print(f'Saved to {HISTOGRAM_FILE} in {t} seconds.')

