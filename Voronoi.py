import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Define base station coordinates (Example: Modify as per your dataset)
base_stations = np.array([
    [2, 3], [5, 8], [9, 6], [4, 4], [7, 2], [6, 9]
])

# Create Voronoi diagram
vor = Voronoi(base_stations)

# Plot Voronoi diagram
fig, ax = plt.subplots(figsize=(8, 8))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.6, point_size=10)

# Plot base stations
ax.scatter(base_stations[:, 0], base_stations[:, 1], c='red', marker='o', edgecolors='black', s=100, label='Base Stations')

# Labels for base stations
for i, (x, y) in enumerate(base_stations):
    ax.text(x, y, f'BS{i+1}', fontsize=12, ha='right', color='blue')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("Voronoi Coverage of Base Stations")
ax.legend()
plt.grid(True)
plt.show()
