"""Plot the descent and vector differentials of various test optimization functions.

Author: Jacob R. Bumgarner
Email: jacobbum21@gmail.com
"""

import numpy as np

from plotting import DescentPlotter
from test_functions import himmelblau_function, six_hump_camel_function, sphere_function

# Make sure to check out the other functions too!

plotter = DescentPlotter(
    sphere_function,
    [-3, 3, -3, 3],
    zscale=10,
    bg_color="black",
    window_size=(2400, 1800),
)

plotter.plot_function(
    cmap="bone",
    show_contours=True,
    contour_line_width=1.5,
)

# Generate a descent path for variou input points
x = np.linspace(-3, 3)
y = np.linspace(-3, 3)
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()

# Add various features to the plotter here for visualization

plotter.generate_gradient_descent_path([x, y], alpha=0.006, verbose=True)
plotter.plot_points([x, y], color="green")
# plotter.plot_point_paths()
plotter.plot_gradient_vectors(point_density=25)

# Or animate a movie with the features added above.
# save_filename = "sphere.mov"
# plotter.animate_point_descent(
#     save_filename,
#     approach_frames=180,
#     show_path_history=False,  # show a line of the history of points
#     fps=60,
#     point_radius=3,
#     path_radius=2,
#     start_color="red",
#     path_color="red",
#     end_color="red",
#     cmap="plasma",
#     render_offscreen=True,
# )

plotter.show()
