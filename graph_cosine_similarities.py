import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

COSINE_SIMILARITIES_FILE = sys.argv[1] if len(sys.argv) > 1 else 'embeddings_cosine_similarities.pt'

print('Loading cosine cosine_similarities...')
t = time.perf_counter()

cosine_similarities = torch.load(COSINE_SIMILARITIES_FILE)

t = time.perf_counter() - t
print(f'Loaded {cosine_similarities.size(0)} cosine similarity values in {t} seconds.')

print('Creating histogram...')
t = time.perf_counter()
histogram, boundaries = torch.histogram(cosine_similarities, bins=512, density=True)
x_values = 0.5 * (boundaries[:-1] + boundaries[1:])

t = time.perf_counter() - t
print(f'Created histogram in {t} seconds.')


mu = torch.mean(cosine_similarities)
sigma = torch.std(cosine_similarities)

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

# Gaussian (PDF)
y = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)

plt.plot(x, y)
plt.fill_between(x, y, alpha=0.3)


plt.plot(x_values, histogram)
plt.fill_between(x_values, histogram, alpha=0.3)
# plt.axvline(torch.mean(cosine_similarities).item(), linestyle='--', color='black')
plt.axvline(0, linestyle='--', color='black')


ax = plt.gca()

ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
ax.xaxis.set_minor_locator(AutoMinorLocator(10))

ax.yaxis.set_major_locator(MaxNLocator(nbins=20))
ax.yaxis.set_minor_locator(AutoMinorLocator(10))

ax.minorticks_on()
ax.grid(True, which='major')
ax.grid(True, which='minor', alpha=0.3)

plt.show()
