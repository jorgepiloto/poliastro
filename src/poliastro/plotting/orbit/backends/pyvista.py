"""A module implementing orbit plotter backends based on PyVista."""

from itertools import cycle

import numpy as np
import pyvista as pv
from pyvista import examples as pv_examples

from poliastro.plotting.orbit.backends._base import OrbitPlotterBackend
from poliastro.plotting.util import generate_sphere


class BasePyVista(OrbitPlotterBackend):
    """An orbit plotter backend class based on PyVista."""

    def __init__(self, pyvista_plotter):
        """Initializes a backend instance.

        Parameters
        ----------
        pyvista_plotter : ~pyvista.Plotter
            The PyVista plotter to render the scene.

        """
        super().__init__(pyvista_plotter, self.__class__.__name__)
        self.pyvista_plotter.enable_anti_aliasing()

    @property
    def pyvista_plotter(self):
        """Returns the PyVista plotter instance use for rendering the scene.

        Returns
        -------
        ~pyvista.Plotter
            The PyVista plotter to render the scene.

        """
        return self.scene

    def _get_colors(self, color, trail):
        """Return the required list of colors if orbit trail is desired.

        Parameters
        ----------
        color : str
            A string representing the hexadecimal color for the point.
        trail : bool
            ``True`` if orbit trail is desired, ``False`` if not desired.

        Returns
        -------
        list[str]
            A list of strings representing hexadecimal colors.

        """
        return [color]

    def undraw_attractor(self):
        """Removes the attractor from the scene."""
        pass

    def draw_position(self, position, *, color, label, size):
        """Draws the position of a body in the scene.

        Parameters
        ----------
        position : list[float, float, float]
            A list containing the x, y and z coordinates of the point.
        color : str, optional
            A string representing the hexadecimal color for the marker.
        size : float, optional
            The size of the marker.
        label : str
            The name shown in the figure legend to identify the position.

        Returns
        -------
        [~plotly.graph_objects.Surface, ~plotly.graph_objects.Trace]
            An object representing the trace of the coordinates in the scene.

        """
        position_mesh = pv.Sphere(radius=size, center=position)
        self.pyvista_plotter.add_mesh(position_mesh, color=color, label=label)
        return position_mesh

    def draw_impulse(self, position, *, color, label, size):
        """Draws an impulse into the scene.

        Parameters
        ----------
        position : list[float, float]
            A list containing the x and y coordinates of the impulse location.
        color : str, optional
            A string representing the hexadecimal color for the impulse marker.
        label : str
            The name shown in the figure legend to identify the impulse.
        size : float, optional
            The size of the marker for the impulse.

        Returns
        -------
        object
            An object representing the trace of the impulse in the scene.

        """
        impulse_mesh = pv.Sphere(radius=size, center=position)
        self.pyvista_plotter.add_mesh(impulse_mesh, color=color, label=label)
        return impulse_mesh

    def update_legend(self):
        """Update the legend of the scene."""
        pass

    def resize_limits(self):
        """Resize the limits of the scene."""
        pass

    def show(self):
        """Displays the scene."""
        self.pyvista_plotter.show()

    def generate_labels(self, label, has_coordinates, has_position):
        """Generates the labels for coordinates and position.

        Parameters
        ----------
        label : str
            A string representing the label.
        has_coordinates : boolean
            Whether the object has coordinates to plot or not.
        has_position : boolean
            Whether the object has a position to plot or not.

        Returns
        -------
        tuple
            A tuple containing the coordinates and position labels.

        """
        return (label, None)


class PyVista3D(BasePyVista):
    """A three-dimensional orbit plotter backend class based on PyVista."""

    def __init__(
            self,
            pyvista_plotter=None,
            use_dark_theme=False,
            use_stars_background=False,
            use_planets_textures=False,
    ):
        """Initializes a backend instance.

        Parameters
        ----------
        pyvista_plotter : ~pyvista.Plotter
            The PyVista plotter to render the scene.
        use_dark_theme : bool, optional
            If ``True``, uses dark theme. If ``False``, uses light theme.
            Default to ``False``.

        """
        # Apply the desired theme
        pyvista_plotter = pyvista_plotter or pv.Plotter()

        # Apply the desired background color
        color = "black" if use_dark_theme is True else "white"
        pyvista_plotter.background_color = color

        # Whether to render or not the stars background
        if use_stars_background is True:
            pyvista_plotter.add_background_image(
                pv_examples.planets.download_stars_sky_background(load=False)
            )

        super().__init__(pyvista_plotter)

    def draw_marker(self, position, *, color, label, marker_symbol, size):
        """Draws a marker into the scene.

        Parameters
        ----------
        position : list[float, float]
            A list containing the x and y coordinates of the point.
        color : str, optional
            A string representing the hexadecimal color for the point.
        label : str
            The name shown in the legend of the figure to identify the marker.
        marker_symbol : str
            The marker symbol to be used when drawing the point.
        size : float, optional
            Desired size for the marker.

        Returns
        -------
        object
            An object representing the trace of the marker in the scene.

        """
        marker_style = dict(size=size, color=color, symbol=marker_symbol)
        marker_trace = go.Scatter(
            x=position[0],
            y=position[1],
            marker=marker_style,
            name=label,
            showlegend=False if label is None else True,
        )
        self.figure.add_trace(marker_trace)
        return marker_trace

    def draw_sphere(self, position, *, color, label, radius):
        """Draws an sphere into the scene.

        Parameters
        ----------
        position : list[float, float]
            A list containing the x and y coordinates of the sphere location.
        color : str, optional
            A string representing the hexadecimal color for the sphere.
        label : str
            Unuseful for this routine. See the ``Notes`` section.
        radius : float, optional
            The radius of the sphere.

        Notes
        -----
        Plotting a sphere in a two-dimensional figure in plotly requires a shape
        instead of a trace. Shapes do not accept a label, as the legend does not
        support labels for shapes.

        Returns
        -------
        dict
            A dictionary representing the shape of the sphere.

        """
        sphere_mesh = pv.Sphere(radius=radius, center=position)
        self.pyvista_plotter.add_mesh(sphere_mesh, color=color, label=label)
        return sphere_mesh

    def draw_coordinates(self, coordinates, *, colors, dashed, label):
        """Draws desired coordinates into the scene.

        Parameters
        ----------
        position : list[list[float, float, float]]
            A set of lists containing the x and y coordinates of the sphere location.
        colors : list[str]
            A list of string representing the hexadecimal color for the coordinates.
        dashed : bool
            Whether to use a dashed or solid line style for the coordiantes.
        label : str
            The name shown in the legend for identifying the coordinates.

        Returns
        -------
        trace_coordinates : object
            An object representing the trace of the coordinates in the scene.

        """
        # Unpack coordinates
        x_coords, y_coords, z_coords = coordinates
        coordinates = np.array([[x, y, z] for x, y, z in zip(x_coords, y_coords, z_coords)])

        # Plot the coordinates in the scene
        if dashed:
            coordinates_mesh = pv.PolyData(coordinates)
        else:
            coordinates_mesh = pv.lines_from_points(coordinates, close=True)

        # Draw the coordinates
        self.pyvista_plotter.add_mesh(
                coordinates_mesh, label=label, color=colors[0]
        )
        return coordinates_mesh

    def draw_axes_labels_with_length_scale_units(self, length_scale_units):
        """Draws the desired label into the specified axis.

        Parameters
        ----------
        lenght_scale_units : ~astropy.units.Unit
            Desired units of lenght used for representing distances.

        """
        # HACK: plotly does not show LaTeX symbols and \text. The usage of
        # ASCII labels in plotly figures is used to avoid this issue
        self.pyvista_plotter.add_axes(
            xlabel=f"x ({length_scale_units.name})",
            ylabel=f"y ({length_scale_units.name})",
            zlabel=f"z ({length_scale_units.name})",
        )
