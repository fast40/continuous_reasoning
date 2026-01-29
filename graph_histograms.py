import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

HISTOGRAM_FILES = sys.argv[1:]

print(f'=== GRAPHING HISTOGRAMS FOR {len(HISTOGRAM_FILES)} FILES ===')
for histogram_file in HISTOGRAM_FILES:
    print(histogram_file)

for histogram_file in HISTOGRAM_FILES:
    data = np.load(histogram_file)

    plt.plot(data['x_values'], data['histogram'], label=histogram_file.split('.')[0])
    plt.fill_between(data['x_values'], data['histogram'], alpha=0.3)


plt.axvline(0, linestyle='--', color='black')

ax = plt.gca()

ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
ax.xaxis.set_minor_locator(AutoMinorLocator(10))

ax.yaxis.set_major_locator(MaxNLocator(nbins=20))
ax.yaxis.set_minor_locator(AutoMinorLocator(10))

ax.minorticks_on()
ax.grid(True, which='major')
ax.grid(True, which='minor', alpha=0.3)

plt.legend()

plt.show()
