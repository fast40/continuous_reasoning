import time
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

COSINE_SIMILARITIES_FILE = sys.argv[1] if len(sys.argv) > 1 else 'embeddings_cosine_similarities.pt'

model = AutoModelForCausalLM.from_pretrained('gpt2')

t = time.perf_counter()

embeddings = model.transformer.wte.weight.detach()
embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
indices = torch.triu_indices(embeddings_normalized.size(0), embeddings_normalized.size(0), offset=1)
similarity_scores = (embeddings_normalized @ embeddings_normalized.T)
similarity_scores = similarity_scores[indices[0], indices[1]]

t = time.perf_counter() - t

print(f'Calculated {similarity_scores.size(0)} cosine similarities in {t} seconds.')

torch.save(similarity_scores, COSINE_SIMILARITIES_FILE)

print(f'Saved to {COSINE_SIMILARITIES_FILE}.')

