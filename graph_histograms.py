import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

HISTOGRAM_FILES = sorted(sys.argv[1:])

print(f'=== GRAPHING HISTOGRAMS FOR {len(HISTOGRAM_FILES)} FILES ===')
for histogram_file in HISTOGRAM_FILES:
    print(histogram_file)

rainbow24 = [
  "#ff0000", "#ff2f00", "#ff6400", "#ff9a00", "#ffcf00", "#f8fd00",
  "#c5ff00", "#90ff00", "#5bff00", "#25ff00", "#02ff0c", "#00ff3f",
  "#00ff74", "#00ffa9", "#00ffde", "#00eaff", "#00b5ff", "#0080ff",
  "#004bff", "#001cff", "#1900ff", "#4f00ff", "#8400ff", "#b900ff",
]

# rainbow24 = [rainbow24[0], rainbow24[5], rainbow24[10]]

for color, histogram_file in zip(rainbow24, HISTOGRAM_FILES):
    data = np.load(histogram_file)

    plt.plot(data['x_values'], data['histogram'], label=histogram_file.split('.')[0], color=color)
    # plt.fill_between(data['x_values'], data['histogram'], alpha=0.3)


plt.title('Density of Pairwise Cosine Similarity Across GPT-2 Coconut Checkpoints')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
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
