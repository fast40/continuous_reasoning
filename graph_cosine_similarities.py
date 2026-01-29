import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

COSINE_SIMILARITIES_FILE = sys.argv[1] if len(sys.argv) > 1 else 'histogram.npz'

data = np.load(COSINE_SIMILARITIES_FILE)

# x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
#
# # Gaussian (PDF)
# y = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)

# plt.plot(x, y)
# plt.fill_between(x, y, alpha=0.3)


plt.plot(data['x_values'], data['histogram'])
plt.fill_between(data['x_values'], data['histogram'], alpha=0.3)
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
