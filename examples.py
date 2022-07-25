"""Plot the descent and vector differentials of various test optimization functions.

Author: Jacob R. Bumgarner
Email: jacobbum21@gmail.com
"""

import numpy as np

from grad_descent_visualizer import DescentPlotter
from grad_descent_visualizer.test_functions import (
    griewank_function,
    six_hump_camel_function,
)


## Example 1 - Plot the descent of a single point on the default Himmelblau Function
plotter = DescentPlotter(
    bg_color="black"
)

plotter.plot_function(
    cmap="viridis_r",
    show_contours=True,
    contour_line_width=1.5,
)  # plot the function

x, y = -0.4, -0.65
plotter.generate_gradient_descent_path([x, y])  # examine the point path
plotter.plot_point_paths()  # show the path
plotter.show()  # visualize the scene

## Example 2 - Plot the gradient vectors of a function
plotter = DescentPlotter(
    test_function=griewank_function,
    axes_ranges=[-4, 4, -4, 4],
    zscale=50,
    bg_color="black",
)  # Create the plotter

plotter.plot_function(cmap="plasma")  # plot the function

# Create the vector grid points
x = y = np.linspace(-3, 3, 20)
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()

plotter.plot_gradient_vectors(
    XY_coordinates=[x, y],
    vector_scalar=30,
    color="red"
)  # plot the grid

plotter.show()  # show the scene

## Example 3 - Generate a Gradient Descent Animation
plotter = DescentPlotter(
    six_hump_camel_function, axes_ranges=[-2, 2, -1, 1], zscale=50, bg_color="black"
)
plotter.plot_function(cmap="plasma_r", show_contours=True, contour_line_width=1.5)

# Generate a descent path for a grid of input points
x = np.linspace(-2, 2, 10)
y = np.linspace(-1, 1, 10)
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()

# Add various features to the plotter here for visualization
plotter.generate_gradient_descent_path([x, y], alpha=0.005, verbose=True)

# Animate a movie with the features added above.
save_filename = "six_camel_descent.mov"
plotter.animate_point_descent(
    save_filename,
    approach_frames=60,  # the number of frames prior to starting the descnet
    buffer_frames=30,  # the number of frames between the approach and start of descent
    show_path_history=True,  # show the history of points
    fps=60,
    point_radius=3,
    path_radius=2,
    cmap="jet"
    # start_color="orange",
    # path_color="red",
    # end_color="green",
)
