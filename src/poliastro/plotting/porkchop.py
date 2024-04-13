"""This is the implementation of porkchop plot."""
from astropy import coordinates as coord, units as u
from matplotlib import pyplot as plt
from matplotlib import patheffects
import numpy as np

from poliastro.bodies import (
    Earth,
    Jupiter,
    Mars,
    Mercury,
    Moon,
    Neptune,
    Pluto,
    Saturn,
    Sun,
    Uranus,
    Venus,
)
from poliastro.maneuver import Maneuver
from poliastro.twobody.orbit import Orbit
from poliastro.util import norm


def _get_state(body, time):
    """Computes the position of a body for a given time."""
    solar_system_bodies = [
        Sun,
        Mercury,
        Venus,
        Earth,
        Moon,
        Mars,
        Jupiter,
        Saturn,
        Uranus,
        Neptune,
        Pluto,
    ]

    # We check if body belongs to poliastro.bodies
    if body in solar_system_bodies:
        rr, vv = coord.get_body_barycentric_posvel(body.name, time)
    else:
        rr, vv = body.propagate(time).rv()
        rr = coord.CartesianRepresentation(rr)
        vv = coord.CartesianRepresentation(vv)

    return rr.xyz, vv.xyz


def _targetting(departure_body, target_body, t_launch, t_arrival):
    """This function returns the increment in departure and arrival velocities."""
    # Get position and velocities for departure and arrival
    rr_dpt_body, vv_dpt_body = _get_state(departure_body, t_launch)
    rr_arr_body, vv_arr_body = _get_state(target_body, t_arrival)

    # Transform into Orbit objects
    attractor = departure_body.parent
    orb_dpt = Orbit.from_vectors(
        attractor, rr_dpt_body, vv_dpt_body, epoch=t_launch
    )
    orb_arr = Orbit.from_vectors(
        attractor, rr_arr_body, vv_arr_body, epoch=t_arrival
    )

    # Define time of flight
    tof = orb_arr.epoch - orb_dpt.epoch

    if tof.to_value(u.s) <= 0:
        return None, None, None, None, None

    try:
        # Lambert is now a Maneuver object
        man_lambert = Maneuver.lambert(orb_dpt, orb_arr)

        # Get norm delta velocities
        dv_dpt = norm(man_lambert.impulses[0][1])
        dv_arr = norm(man_lambert.impulses[1][1])

        # Compute all the output variables
        c3_launch = dv_dpt**2
        c3_arrival = dv_arr**2

        return (
            dv_dpt.to_value(u.km / u.s),
            dv_arr.to_value(u.km / u.s),
            c3_launch.to_value(u.km**2 / u.s**2),
            c3_arrival.to_value(u.km**2 / u.s**2),
            tof.jd,
        )

    except AssertionError:
        return None, None, None, None, None


# numpy.vectorize is amazing
targetting_vec = np.vectorize(
    _targetting,
    otypes=[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    excluded=[0, 1],
)


class PorkchopPlotter:
    """Class Implementation for Porkchop Plot.

    Parameters
    ----------
    departure_body: poliastro.bodies.Body
        Body from which departure is done.
    target_body: poliastro.bodies.Body
        Body for targetting.
    launch_span: astropy.time.Time
        Time span for launch.
    arrival_span: astropy.time.Time
        Time span for arrival.
    ax: matplotlib.axes.Axes
        For custom figures.
    
    """
    def __init__(
        self,
        departure_body,
        target_body,
        launch_span,
        arrival_span,
        ax=None,
    ):
        self.departure_body = departure_body
        self.target_body = target_body
        self.launch_span = launch_span
        self.arrival_span = arrival_span
        self.ax = ax

    def plot(
        self, 
        plot_c3_lines=True,
        plot_tof_lines=True,
        plot_dv_lines=True,
        plot_av_lines=False,
        c3_levels=np.linspace(0, 45, 30) * u.km**2 / u.s**2,
        tof_levels=np.linspace(0, 500, 5) * u.d,
        dv_levels=np.linspace(0, 5, 5) * u.km / u.s,
        av_levels=np.linspace(0, 5, 5) * u.km / u.s,
        title=None,
    ):
        """Plots porkchop between two bodies.

        Parameters
        ----------
        plot_c3_lines : bool
            For plotting C3 contour lines.
        plot_tof_lines : bool
            For plotting time flight contour lines.
        plot_dv_lines : bool
            For plotting departure velocity contour lines.
        plot_av_lines : bool
            For plotting arrival velocity contour lines.
        c3_levels: numpy.ndarray
            Levels for c3 contour lines.
        dvl_levels: numpy.ndarray
            Levels for departure velocity contour lines.
        avl_levels: numpy.ndarray
            Levels for arrival velocity contour lines.
        title : str
            Title of the plot.

        Returns
        -------
        dv_launch: numpy.ndarray
            Launch delta v
        dv_arrival: numpy.ndarray
            Arrival delta v
        c3_launch: numpy.ndarray
            Characteristic launch energy
        c3_arrrival: numpy.ndarray
            Characteristic arrival energy
        tof: numpy.ndarray
            Time of flight for each transfer

        Examples
        --------
        >>> from poliastro.plotting.porkchop import PorkchopPlotter
        >>> from poliastro.bodies import Earth, Mars
        >>> from poliastro.util import time_range
        >>> launch_span = time_range("2005-04-30", end="2005-10-07")
        >>> arrival_span = time_range("2005-11-16", end="2006-12-21")
        >>> porkchop_plot = PorkchopPlotter(Earth, Mars, launch_span, arrival_span)
        >>> dv_launch, dev_dpt, c3dpt, c3arr, tof = porkchop_plot.porkchop()

        """
        self.c3_levels = c3_levels
        self.tof_levels = tof_levels
        self.dv_levels = dv_levels
        self.av_levels = av_levels
        self.plot_c3_lines = plot_c3_lines
        self.plot_tof_lines = plot_tof_lines
        self.plot_dv_lines = plot_dv_lines
        self.plot_av_lines = plot_av_lines

        # Compute porkchop values
        dv_launch, dv_arrival, c3_launch, c3_arrival, tof = targetting_vec(
            self.departure_body,
            self.target_body,
            self.launch_span[np.newaxis, :],
            self.arrival_span[:, np.newaxis],
        )

        # Start drawing porkchop
        if self.ax is None:
            fig, self.ax = plt.subplots(figsize=(15, 15))
        else:
            fig = self.ax.figure

        # Draw the contour with colors and the colorbar
        c3_colors = self.ax.contourf(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            c3_launch.astype("float64"),
            c3_levels.astype("float64"),
        )
        c3_colorbar = fig.colorbar(c3_colors)
        c3_colorbar.set_label("$km^2 / s^2$")

        # Draw the solid contour lines on top of previous colors
        if self.plot_c3_lines:
            c3_lines = self.ax.contour(
                [D.to_datetime() for D in self.launch_span],
                [A.to_datetime() for A in self.arrival_span],
                c3_launch.astype("float64"),
                c3_levels.astype("float64"),
                colors="black",
                linestyles="solid",
            )
            self.ax.clabel(c3_lines, inline=1, fmt="%1.1f", colors="k", fontsize=10)

        # Draw the time of flight lines (if requested)
        if self.plot_tof_lines:
            tof_lines = self.ax.contour(
                [D.to_datetime() for D in self.launch_span],
                [A.to_datetime() for A in self.arrival_span],
                (tof / 365.25).astype("float64"),
                self.tof_levels.to_value(u.year).astype("float64"),
                colors="red",
                linestyles="dashed",
                linewidths=3.5,
            )
            #tof_lines.set(path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
            tof_lines_labels = self.ax.clabel(
                tof_lines, inline=1, fmt="%1.0f years", colors="r", fontsize=14, use_clabeltext=True
            )
            #plt.setp(tof_lines_labels, path_effects=[
            #    patheffects.withStroke(linewidth=3, foreground="k")])

        # Draw the departure velocity lines (if requested)
        if self.plot_dv_lines:
            dvl_lines = self.ax.contour(
                [D.to_datetime() for D in self.launch_span],
                [A.to_datetime() for A in self.arrival_span],
                dv_launch.astype("float64"),
                self.dv_levels.astype("float64"),
                colors="red",
                linewidths=2.0,
            )
            self.ax.clabel(
                dvl_lines, inline=1, fmt="%1.1f", colors="red", fontsize=12
            )

        # Draw the arrival velocity lines (if requested)
        if self.plot_av_lines:
            avl_lines = self.ax.contour(
                [D.to_datetime() for D in self.launch_span],
                [A.to_datetime() for A in self.arrival_span],
                dv_arrival.astype("float64"),
                self.av_levels.astype("float64"),
                colors="white",
                linewidths=2.0,
            )
            avl_lines.set(path_effects=[patheffects.withStroke(linewidth=3, foreground="k")])
            avl_labels = self.ax.clabel(
                avl_lines, inline=1, fmt="%1.0f km/s", colors="white",
                fontsize=14, use_clabeltext=True
            )
            plt.setp(avl_labels, path_effects=[
                patheffects.withStroke(linewidth=3, foreground="k")])

        if title:
            self.ax.set_title(title, fontsize=14, fontweight="bold")

        self.ax.set_xlabel("Launch date", fontsize=10, fontweight="bold")
        self.ax.set_ylabel("Arrival date", fontsize=10, fontweight="bold")

        # Plot the minimum C3 launch energy point
        min_c3 = c3_launch.min()        
        launch_date_at_c3_min = np.meshgrid(self.launch_span, self.arrival_span)[0][np.unravel_index(c3_launch.argmin(), c3_launch.shape)]
        arrival_date_at_c3_min = np.meshgrid(self.launch_span, self.arrival_span)[1][np.unravel_index(c3_launch.argmin(), c3_launch.shape)]


        self.ax.plot(
                launch_date_at_c3_min.to_datetime(),
                arrival_date_at_c3_min.to_datetime(),
                "ko",
                markersize=15,
        )
        print(f"MINIMUM C3: {min_c3}")
        print(f"LAUNCH AT: {launch_date_at_c3_min}")
        print(f"ARRIVAL AT: {arrival_date_at_c3_min}")



        return (
            dv_launch * u.km / u.s,
            dv_arrival * u.km / u.s,
            c3_launch * u.km**2 / u.s**2,
            c3_arrival * u.km**2 / u.s**2,
            tof * u.d,
        )
