"""The plotter class for function and descent visualization."""
import warnings
from time import perf_counter as pf
from typing import Callable, Sequence, Union

import numpy as np
import pyvista as pv

from . import descent
from .test_functions import himmelblau_function


class DescentPlotter:
    """A plotter for visualizing test_function landscapes and gradient descents.

    Args:
        test_function(Callable, optional): The test function used for plotting and
            gradient descent. Pre-loaded test_functions can be found in
            test_functions.py. Defaults to himmelblau_function.
        axes_ranges (Sequence[float], optional): The minimum and maximum ranges of
            the axes. Should be passed in the format [xmin, xmax, ymin, ymax].
            Defaults to [-5, 5, -4, 3.5].
        zscale (float, optional): The amount by which to scale the Z dimension.
            Defaults to 1 for the Himmelblau function.
            X and Y linearly spaced arrays to plot the grid. Defaults to 1000.
        grid_plotting_resolution (int, optional): The number of points used to create
        window_size (Sequence[int]): The size of the plotter.
        bg_color (str): The background color of the plotter. Defaults to False.

    Params:
        actors (list): A list of lists. Each sublist contains a mesh and a dict of the
            kwargs used to add the mesh to the plotter. Used to restore the plotter
            when creating a movie.
    """

    def __init__(
        self,
        test_function: Callable = himmelblau_function,
        axes_ranges: Sequence[float] = [-5, 5, -4.5, 3.5],
        zscale: float = 1,
        grid_plotting_resolution: int = 500,
        window_size: Sequence[int] = (1200, 1200),
        bg_color: str = "#4c4c4c",
    ) -> None:
        """Create the descent plotter."""
        self.test_function = test_function
        self.descent_path = None

        self.axes_ranges = axes_ranges
        self.grid_plotting_resolution = grid_plotting_resolution
        self.zscale = zscale

        self.window_size = window_size
        self.bg_color = bg_color
        self.create_plotter()

        self.actors = []
        return

    def create_plotter(self, off_screen: bool = False) -> None:
        """Create a plotter from the input settings.

        Args:
            off_screen(bool, optional): Render the plotter off screen. Defaults to
                False.

        """
        self.plotter = pv.Plotter(window_size=self.window_size, off_screen=off_screen)
        self.plotter.set_background(self.bg_color)

    @property
    def test_function(self):
        """The test_function attribute."""
        return self._test_function

    @test_function.setter
    def test_function(self, test_function: Callable):
        """The setter for the test_function attribute."""
        if isinstance(test_function, Callable):
            self._test_function = test_function
        else:
            raise TypeError(
                "The test function must be a function with an equation.\n"
                "See test_functions.py for example test functions."
            )

    def prepare_coordinates(
        self,
        X: Union[float, np.ndarray],
        Y: Union[float, np.ndarray],
        Z: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Prepare the X, Y, and Z variables for plotting.

        Args:
            X (Union[float, np.ndarray]): The X coordinates
            Y (Union[float, np.ndarray]): The Y coordinates
            Z (Union[float, np.ndarray]): The Z coordinates

        Returns:
            np.ndarray: The shaped array.
        """
        if isinstance(Z, (float, int)):
            coords = np.array([[X, Y, Z]])
        else:
            coords = np.stack([X, Y, Z], axis=1)
        return coords

    def scale_coordinates(
        self,
        coords: np.ndarray,
    ) -> np.ndarray:
        """Scale an array of coordinates into a matching grid resolution.

        Args:
            coords (np.ndarray): The input XYZ coordinates
            axes_ranges (Sequence[float]): The XY axes ranges.
            resolution (int): The element-wise resolution of the grid.
            zscale (float): The z-scaling of the grid.

        Returns:
            np.ndarray: The scaled array of coordinates.
        """
        # Convert the coordinates into the grid space using the axis ranges & resolution
        # find the distance from the minima
        coords[:, :2] -= [self.axes_ranges[0], self.axes_ranges[2]]
        coords[:, :2] = np.abs(coords[:, :2])

        # convert distance into 0-1 scale
        coords[:, :2] /= [
            np.abs(self.axes_ranges[0] - self.axes_ranges[1]),
            np.abs(self.axes_ranges[2] - self.axes_ranges[3]),
        ]

        # scale coordinates to the grid
        coords *= [
            self.grid_plotting_resolution,
            self.grid_plotting_resolution,
            self.zscale,
        ]

        return coords

    def polyline_from_points(self, points: np.ndarray) -> pv.PolyData:
        """Generate a polyline from a set of 3D points.

        Args:
            points (np.ndarray): The sequential series of 3D points in (m, n) shape.

        Returns:
            pv.PolyData: The polyline mesh.
        """
        polydata = pv.PolyData()
        polydata.points = points
        the_cell = np.arange(0, len(points), dtype=np.int_)
        the_cell = np.insert(the_cell, 0, len(points))
        polydata.lines = the_cell
        return polydata

    def generate_pd_colormap(
        self, n_colors: int, randomize_colormap: bool
    ) -> pv.PolyData:
        """Generate a colormap scalar for a series of input coordinates.

        Args:
            n_colors (int): The number of colors.
            randomize_colormap (bool): _description_

        Returns:
            pv.PolyData: _description_
        """
        if randomize_colormap:
            colors = np.random.choice(np.arange(0, n_colors), n_colors, replace=False)
        else:
            colors = np.arange(0, n_colors)

        return colors

    def plot_function(
        self,
        cmap: str = "viridis",
        clim: list = None,
        color: str = None,
        show_scalar_bar: bool = False,
        show_contours: bool = False,
        contour_color: str = "f7f7f7",
        contour_line_width: float = 3,
        save_actor: bool = True,
    ) -> None:
        """Plot an input function at a specified resolution.

        Args:
            clim (list, optional): The colormap for the function. Defaults to None.
            color (str, optional): The optional color of the plane. Overrides cmap if
                passed. Defaults to None.
            show_scalar_bar(bool): Whether to show the scalar bar.
                Defaults to False.
            show_contours (bool, optional): Whether to show contours of the function.
                Defaults to False.
            contour_color (str, optional): The color of the contour lines.
                Defaults to "white".
            contour_line_width (float, optional): The width of the contour lines.
                Defaults to 5.
            save_actor (bool, optional): Whether to save the mesh and kwargs used to
                generate the actor. Defaults to True.
        """
        X = np.linspace(
            self.axes_ranges[0], self.axes_ranges[1], self.grid_plotting_resolution
        )
        Y = np.linspace(
            self.axes_ranges[2], self.axes_ranges[3], self.grid_plotting_resolution
        )
        X, Y = np.meshgrid(X, Y)  # prepare the X, Y grids for the function computation

        Z = self.test_function(X, Y) * self.zscale  # compute the function and scale
        Z = np.expand_dims(Z, axis=-1)

        # Plot the function using PyVista
        grid = pv.UniformGrid(dims=Z.shape)
        grid["elevation"] = Z.ravel()
        grid = grid.warp_by_scalar("elevation")

        # Prepare the color
        if color is not None:
            cmap = None

        kwargs = {
            "cmap": cmap,
            "show_scalar_bar": show_scalar_bar,
            "smooth_shading": False,
            "clim": clim,
            "color": color,
        }
        self.plotter.add_mesh(grid, **kwargs)
        if save_actor:
            self.actors.append([grid, kwargs])

        if show_contours:
            contours = grid.contour()
            kwargs = {"color": contour_color, "line_width": contour_line_width}
            self.plotter.add_mesh(contours, **kwargs)

            if save_actor:
                self.actors.append([contours, kwargs])

        return

    def plot_gradient_vectors(
        self,
        XY_coordinates: Union[Sequence[np.ndarray], np.ndarray] = None,
        point_density: int = 50,
        vector_scalar: float = 20,
        color: str = "red",
        cmap: str = None,
        gradient_cmap: bool = True,
        randomize_colormap: bool = True,
        save_actor: bool = True,
    ):
        """Plot vectors showing the gradients of points on a function.

        There are two options for this function. A list of XY points can be passed, or
        a uniform grid of points will .

        Args:
            XY_coordinates (Sequence[np.ndarray], np.ndarray): The coordinates should be
                passed such that the X and Y variables are separate elements of a list.
                Defaults to None.
            point_density(int): The density of the gradients to examine on the function.
                Defaults to 50.
            axes_ranges (Sequence[float]): The XY ranges of the plotted function.
            resolution (int): The resolution of the plotted function as an integer.
            vector_scalar (float): The scalar used to size the vectors.
            color (str, optional): _description_. Defaults to "red".
            cmap (str, optional): The colormap for the vectors. If a colormap is passed,
                the vectors will be colored by the largest gradient of the X or Y
                direction. Defaults to None.
            gradient_cmap (bool, optional): Indicates whether the colormap should show
                the max XY gradient. Only relevant when cmap is not None.
                Defaults to True.
            randomize_colormap (bool, optional): If True and cmap is not None,
                the colors of the points are randomized by the point plots.
            save_actor (bool, optional): Indicates whether to save the mesh and kwargs
                used to generate the actor. Defaults to True.
        """
        # Prepare the points to plot
        if isinstance(XY_coordinates, list):
            X = XY_coordinates[0]
            Y = XY_coordinates[1]
        else:
            X = np.linspace(self.axes_ranges[0], self.axes_ranges[1], point_density)
            Y = np.linspace(self.axes_ranges[2], self.axes_ranges[3], point_density)
            X, Y = np.meshgrid(X, Y)
            X, Y = X.ravel(), Y.ravel()

        Z = self.test_function(X, Y)
        unscaled_coords = self.prepare_coordinates(X, Y, Z)

        coords = self.scale_coordinates(unscaled_coords.copy())
        ### Glyphing Vectors
        differentials = descent.differentiate(self.test_function, element_wise=True)

        vectors = unscaled_coords.copy()
        new_XY = descent.descend_gradient(
            differentials, unscaled_coords[:, :2].T, alpha=0.001
        )
        vectors[:, :2] = np.array(new_XY).T
        vectors[:, 2] = self.test_function(vectors[:, 0], vectors[:, 1])

        vectors = self.scale_coordinates(vectors)
        vectors -= coords
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-5

        polydata = pv.PolyData(coords)
        polydata["vectors"] = vectors

        if cmap is not None:
            if gradient_cmap:
                grad_x = differentials[0](*unscaled_coords[:, :2].T)
                grad_y = differentials[1](*unscaled_coords[:, :2].T)
                grads = np.stack([grad_x, grad_y], axis=1)
                colors = np.max(np.abs(grads), axis=1)
            else:
                colors = self.generate_pd_colormap(coords.shape[0], randomize_colormap)
            polydata["colors"] = colors

        scalars = "colors" if cmap is not None else None

        polydata["scalars"] = np.full(vectors.shape[0], vector_scalar)
        arrow_glyph = polydata.glyph(geom=pv.Arrow(), orient="vectors", scale="scalars")

        kwargs = {
            "color": color,
            "cmap": cmap,
            "scalars": scalars,
            "smooth_shading": True,
            "show_scalar_bar": False,
        }
        actor = self.plotter.add_mesh(arrow_glyph, **kwargs)

        if save_actor:
            self.actors.append([arrow_glyph, kwargs])

        return actor

    def plot_points(
        self,
        coords: Union[Sequence[float], np.ndarray],
        radius: float = 2,
        color: str = "red",
        cmap: str = None,
        colors: np.ndarray = None,
        randomize_colormap: bool = True,
        save_actor: bool = True,
    ):
        """Plot a point as a sphere.

        Args:
            coords (Union[Sequence[float], np.ndarray]): The XY coordinates
                to be plotted. Should be passed as a list of two numbers or a 2D array
                representing the X and Y coordinates of the points.
            radius (float, optional): The reaiuds of the sphere. Defaults to 2.
            color (str, optional): The color of the sphere. Defaults to "red".
            resolution (int, optional): The amount of points to be evaluated. Defaults to
                1000.
            cmap (str, optional): The colormap for the spheres. If a colormap is passed,
                the spheres will be colored by the colormap. Defaults to None.
            colors (np.ndarray, optional): The numbers used to color the points from the
                colormap. Defaults to None.
            randomize_colormap (bool, optional): If True and cmap is not None,
                the colors of the points are randomized by the point plots.
            save_actor (bool, optional): Indicates whether to save the generated actor.
                Defaults to True.

        Returns:
            vtk.vtkActor
                Returns the actor of the added sphere.
        """
        # Convert coords into array
        if not isinstance(coords, (list, np.ndarray)) or len(coords) != 2:
            raise TypeError(
                "The coords argument must be passed as a list of two floats, a list of "
                "two arrays, or a (2, m) shaped np.array."
            )

        Z = self.test_function(coords[0], coords[1])

        coords = self.prepare_coordinates(coords[0], coords[1], Z)

        coords = self.scale_coordinates(coords)

        polydata = pv.PolyData(coords)

        if cmap is not None:
            color = None
            if colors is None:
                colors = self.generate_pd_colormap(coords.shape[0], randomize_colormap)
            polydata["colors"] = colors

        sphere_glyph = polydata.glyph(
            geom=pv.Sphere(radius=radius, theta_resolution=15, phi_resolution=15),
            orient=False,
            scale=False,
        )

        kwargs = {
            "color": color,
            "cmap": cmap,
            "smooth_shading": True,
            "show_scalar_bar": False,
        }
        actor = self.plotter.add_mesh(sphere_glyph, **kwargs)

        if save_actor:
            self.actors.append([sphere_glyph, kwargs])

        return actor

    def plot_point_paths(
        self,
        descent_path: np.ndarray = None,
        radius: float = 2,
        color: str = "red",
        cmap: str = None,
        colors: np.ndarray = None,
        randomize_colormap: bool = True,
        save_actor: bool = True,
        verbose: bool = False,
    ):
        """Plot the paths of a point on the input function.

        Args:
            descent_path (np.ndarray, optional): The (m, 3, n) path of the points to
                plot. If None, uses the generated descent path. Defaults to None.
            radius (float, optional): The reaiuds of the sphere. Defaults to 2.
            color (str, optional): The color of the sphere. Defaults to "red".
            resolution (int, optional): The amount of points to be evaluated. Defaults
                to 1000.
            cmap (str, optional): The colormap for the spheres. If a colormap is passed,
                the spheres will be colored by the colormap. Defaults to None.
            colors (np.ndarray, optional): The numbers used to color the points from the
                colormap. Defaults to None.
            randomize_colormap (bool, optional): If True and cmap is not None.
            save_actor (bool, optional): Indicates whether to save the generated actor.
                Defaults to True.
            verbose (bool, optional): Print the status of the plotting.
                Defaults to False.

        """
        if descent_path is None and self.descent_path is None:
            raise ValueError(
                "Either a descent_path must be passed or a descent_path must be "
                "generated with `DescentPlotter.generate_descent_path`."
            )

        paths = descent_path if descent_path is not None else self.descent_path
        paths = paths.T

        # Prep the colors array
        if cmap is not None:
            color = None
            if colors is None:
                colors = self.generate_pd_colormap(paths.shape[0], randomize_colormap)

        # Plot the point paths
        path_multiblock = pv.MultiBlock()
        for i in range(paths.shape[0]):
            if verbose:
                print(f"Plotting path: {i+1}/{paths.shape[0]}", end="\r")

            point_path = self.scale_coordinates(paths[i].T)
            polydata = self.polyline_from_points(point_path)

            if colors is not None:
                polydata["color"] = np.full(point_path.shape[0], colors[i])

            line = polydata.tube(radius=radius, n_sides=4)

            path_multiblock.append(line)

        path_multiblock = path_multiblock.combine()

        kwargs = {
            "color": color,
            "cmap": cmap,
            "smooth_shading": True,
            "show_scalar_bar": False,
        }

        actor = self.plotter.add_mesh(path_multiblock, **kwargs)

        if save_actor:
            self.actors.append([path_multiblock, kwargs])

        return actor

    def check_frame_count(self, frames: int, path_steps: int) -> int:
        """Return the appropriate number of frames for the movie.

        Args:
            frames (int): The proposed frame count.
            path_steps (int): The number of steps in the descent path.

        Returns:
            int: max(frames, path_steps)
        """
        if frames < path_steps:
            warnings.warn(
                "Warning: The input frames is smaller than the descent iterations.\n"
                f"Instead of {frames} descent frames, there will be {path_steps} "
                "descent frames in the movie."
            )
            frames = path_steps
        return frames

    def animate_point_descent(
        self,
        save_path: str,
        approach_frames: int = 60,
        buffer_frames: int = 30,
        descent_frames: int = 150,
        fps: int = 30,
        descent_path: np.ndarray = None,
        show_path_history: bool = True,
        point_radius: float = 2,
        path_radius: float = 2,
        start_color: str = "red",
        path_color: str = "red",
        end_color: str = "red",
        cmap: str = None,
        randomize_colormap: bool = True,
        render_offscreen: bool = True,
        verbose: bool = True,
    ):
        """Plot the paths of a point on the input function.

        Args:
            save_path (str): The save path for the output movie.
            approach_frames (int, optional): The number of camera movement frames prior
                to the onset of the descent. Defaults to 0.
            buffer_frames (int, optional): The number of frames between the approach
                and start of the descent. Only applies if approach_frames != 0. Defaults
                to 30.
            descent_frames (int, optional): The number of movie frames during the
                descent. Defaults to 150.
            fps (int, optional): The fps of the movie. Defaults to 30.
            descent_path (np.ndarray, optional): The (m, 3, n) path of the points to
                plot. If None, uses the generated descent path. Defaults to None.
            show_path (bool, optional): Plot the history of the path of points. Defaults
                to True.
            point_radius (float, optional): The radius of the spheres. Defaults to 2.
            path_radius (float, optional): The radius of the path tubes. Defaults to 2.
            start_color (str, optional): The color of the starting points. Defaults to
                "red".
            path_color (str, optional): The color of the sphere. Defaults to "red".
            end_color (str, optional): The color of the end points. Defaults to "red".
            resolution (int, optional): The amount of points to be evaluated. Defaults
                to 1000.
            cmap (str, optional): The colormap for the spheres. If a colormap is passed,
                the start_color and path_color will be ignored. Defaults to None.
            randomize_colormap (bool, optional): If True and cmap is not None,
            render_offscreen (bool, optional): Renders the movie offscreen. Defaults to
                True.
            verbose (bool, optional): Print the status of the plotting.
                Defaults to True.
        """
        if descent_path is None and self.descent_path is None:
            raise ValueError(
                "Either a descent_path must be passed or a descent_path must be "
                "generated with `DescentPlotter.generate_descent_path` prior to "
                "creating a movie."
            )

        paths = descent_path if descent_path is not None else self.descent_path

        # Check the frame count
        descent_frames = self.check_frame_count(descent_frames, paths.shape[0])
        total_frames = approach_frames + buffer_frames + descent_frames

        # Prep the colors array
        if cmap is not None:
            point_colors = [None for _ in range(paths.shape[0])]
            colors = self.generate_pd_colormap(paths.shape[-1], randomize_colormap)
        else:
            point_colors = (
                [start_color]
                + [path_color for _ in range(paths.shape[0] - 2)]
                + [end_color]
            )
            colors = None

        # Generate the keyframes. Show the paths to help with keyframe selection.
        self.plot_point_paths(
            paths.copy(),
            radius=path_radius,
            color=path_color,
            cmap=cmap,
            colors=colors,
            save_actor=False,
        )
        self.generate_movie_keyframes(
            approach_frames, descent_frames, offscreen=render_offscreen
        )

        # With the new keyframes, create the movie.
        self.plotter.open_movie(save_path, framerate=fps, quality=9)

        # Get the frame_factor, which shows which frames to update the plotter
        path_update_divisor = self.camera_path.shape[0] // paths.shape[0]

        path_actor = point_actor = None

        # Plot the movie path
        # First plot the approach path, show the starting points
        if approach_frames > 0:
            point_actor = self.plot_points(
                paths[0, :2],
                radius=point_radius,
                color=point_colors[0],
                cmap=cmap,
                colors=colors,
                save_actor=False,
            )
            for i in range(approach_frames):
                if verbose:
                    print(
                        f"Plotting frame: {i+1}/{total_frames}",
                        end="\r",
                    )
                self.plotter.camera_position = self.approach_path[i]
                self.plotter.write_frame()

            for i in range(buffer_frames):
                if verbose:
                    print(
                        f"Plotting frame: {i+1 + approach_frames}/{total_frames}",
                        end="\r",
                    )
                self.plotter.write_frame()

        frame_times = 0
        for i in range(self.camera_path.shape[0]):
            t = pf()
            if verbose:
                remaining_time = frame_times / (i + 1) * (descent_frames - i)
                print(
                    f"Plotting frame: {i+1 + approach_frames + buffer_frames}/{total_frames} "
                    f"Est. remaining time: {remaining_time:0.2f}      ",
                    end="\r",
                )

            # Update the plotter camera position
            self.plotter.camera_position = self.camera_path[i]

            path_index = int(i / path_update_divisor)
            if i % path_update_divisor == 0 and path_index < paths.shape[0]:
                # Remove the previous actors
                self.plotter.remove_actor(point_actor)
                self.plotter.remove_actor(path_actor)

                # First plot the points
                point_actor = self.plot_points(
                    paths[path_index, :2],
                    radius=point_radius,
                    color=point_colors[path_index],
                    cmap=cmap,
                    colors=colors,
                    save_actor=False,
                )

                # Then plot the path
                if show_path_history and path_index > 1:
                    subpath = paths[: path_index + 1].copy()
                    path_actor = self.plot_point_paths(
                        subpath,
                        radius=path_radius,
                        color=path_color,
                        cmap=cmap,
                        colors=colors,
                        randomize_colormap=randomize_colormap,
                        save_actor=False,
                    )

            # Write the frame
            self.plotter.write_frame()

            frame_times += pf() - t

        self.plotter.close()

        return

    def generate_movie_keyframes(
        self, approach_frames: int, descent_frames: int, offscreen: bool
    ) -> None:
        """Select the start and stop keyframes for the movie.

        Args:
            approach_keyframes(int): The number of frames to use in the approach.
            descent_frames (int): The number of frames to interpolate between the
                start/end keyframes.
            offscreen (bool): Whether to open the new plotter off screen.
        """

        def set_approach_frame():
            print("Approach keyframe set!")
            self.approach_keyframe = self.plotter.camera_position

        def set_start_frame():
            print("Start keyframe set!")
            self.start_keyframe = self.plotter.camera_position

        def set_end_frame():
            print("End keyframe set!")
            self.end_keyframe = self.plotter.camera_position

        # define our keyframes
        self.approach_keyframe = self.plotter.camera_position
        self.start_keyframe = self.plotter.camera_position
        self.end_keyframe = self.plotter.camera_position

        # Add the descent path start/stop hotkeys to the plotter
        if approach_frames > 0:
            self.plotter.add_key_event("0", set_approach_frame)
        self.plotter.add_key_event("1", set_start_frame)
        self.plotter.add_key_event("2", set_end_frame)

        if approach_frames > 0:
            self.plotter.add_text("Press '0' to set the approach keyframe", (10, 130))
        self.plotter.add_text("Press '1' to set the start keyframe", (10, 90))
        self.plotter.add_text("Press '2' to set the end keyframe", (10, 50))
        self.plotter.add_text("Press 'q' to render the movie", (10, 10))

        # Show the plotter to prompt the keyframe selection
        self.plotter.show()

        if approach_frames > 0:
            self.approach_path = np.linspace(
                self.prep_keyframe(self.approach_keyframe),
                self.prep_keyframe(self.start_keyframe),
                approach_frames,
                endpoint=True,
            )

        # Create the camera path from the selected keyframes
        self.camera_path = np.linspace(
            self.prep_keyframe(self.start_keyframe),
            self.prep_keyframe(self.end_keyframe),
            descent_frames,
            endpoint=True,
        )

        # Add the actors back to a new plotter
        self.create_plotter(off_screen=offscreen)
        self.restore_actors()

        return

    def restore_actors(self) -> None:
        """Restore previously added actors to the plotter."""
        for mesh, kwargs in self.actors:
            self.plotter.add_mesh(mesh, **kwargs)

    def prep_keyframe(self, position: pv.CameraPosition) -> np.ndarray:
        """Convert a CameraPosition into a numpy array.

        Args:
            position (pv.CameraPosition): The plotter position.

        Returns:
            np.ndarray: The converted np.ndarray
        """
        position = np.array([pos for pos in position])
        return position

    def generate_gradient_descent_path(
        self,
        variables: Sequence[float],
        alpha: float = 0.01,
        tolerance: float = 0.01,
        max_iteration: int = 1000,
        verbose: bool = False,
    ) -> Sequence[Sequence[float]]:
        """Return the path of gradient descent for an input function and input values.

        Args:
            variables (Sequence[float], optional): The input variables to the function.
                The number of variables must match the argument count for the equation.
                Defaults to [0, 0].
            alpha (float, optional): The learning rate for the descent. Defaults to 0.01.
            tolerance (float, optional): The tolerance for the minimum gradient that
                ends the descent. Defaults to 0.001.
            max_iteration (int, optional): The maximum iteration for the descent.
                Defaults to 1000.
            verbose(bool, optional): Defaults to False.

        Returns:
            np.ndarray: An array of the history of the variables. Arranged in
                (m, 3, n) order, where:
                    m - The iterations of descent.
                    3 - the X, Y, Z coordinates of the points.
                    n - The history of a single point
        """
        if verbose:
            t = pf()

        variables = descent.prep_array_for_descent(variables)

        variable_history = [
            descent.save_descent_snapshot(variables, self.test_function)
        ]
        differentials = descent.differentiate(
            self.test_function, element_wise=variables.ndim > 1
        )

        i = 0

        while (
            np.any(
                np.abs(descent.compute_gradients(differentials, variables)) > tolerance
            )
            and i < max_iteration
        ):
            if verbose:
                print(f"Iteration: {i}", end="\r")

            variables = descent.descend_gradient(differentials, variables, alpha)

            variable_history.append(
                descent.save_descent_snapshot(variables, self.test_function)
            )

            i += 1

        if verbose:
            print(f"Function optimized in {i} steps. Total time: {pf() - t: 0.2f}")

        self.descent_path = np.asarray(variable_history)
        return self.descent_path

    def show(self):
        """Presents the plotter."""
        self.plotter.show()
