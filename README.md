# Grad-Descent-Visualizer
A small package used to visualize test function landscapes and gradient descent.

This package was highlighted in my [Medium post](https://medium.com/@jacobbumgarner/breaking-it-down-gradient-descent-b94c124f1dfd) on gradient descent.

<video src="https://user-images.githubusercontent.com/70919881/180077858-14bd8b91-c189-4e52-80d4-332ae5ca2db4.mov"></video>


## Install
```shell-session
pip install grad-descent-visualizer
```

## Usage
**Example 1: Plotting an Example Function Landscape**
```python
from grad_descent_visualizer import DescentPlotter

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
```
<img width="50%" alt="Example 1" src="https://user-images.githubusercontent.com/70919881/180888041-69bc81b1-4071-4616-823a-c8e74dcaf3ae.png">

**Example 2: Plot the gradient vectors of a function**
```python
import numpy as np
from grad_descent_visualizer import DescentPlotter
from grad_descent_visualizer.test_functions import griewank_function

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
```
<img width="50%" alt="Example 2" src="https://user-images.githubusercontent.com/70919881/180888093-6a3707cf-d031-424c-bd92-e2a0e22660ff.png">


**Example 3 - Generate a Gradient Descent Animation**
```python
import numpy as np
from grad_descent_visualizer import DescentPlotter
from grad_descent_visualizer.test_functions import six_camel_hump_function
plotter = DescentPlotter(
    six_hump_camel_function,
    axes_ranges=[-2, 2, -1, 1],
    zscale=50,
    bg_color="black"
)
plotter.plot_function(
    cmap="plasma_r",
    show_contours=True, 
    contour_line_width=1.5)

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

```

<img width="50%" alt="Example 3" src="https://user-images.githubusercontent.com/70919881/180888105-9e0b8aa5-74c6-4cb7-a648-1935ac98ab5b.png">

