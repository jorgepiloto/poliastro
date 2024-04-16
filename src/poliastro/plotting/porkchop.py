"""This is the implementation of porkchop plot."""
from multiprocessing import Pool

from astropy import coordinates as coord, units as u
from matplotlib import pyplot as plt
from matplotlib import patheffects
import numpy as np

from poliastro.bodies import Sun
from poliastro.maneuver import Maneuver
from poliastro.twobody.orbit import Orbit
from poliastro.util import norm


def _targetting(departure_body, target_body, t_launch, t_arrival, prograde):
    """This function returns the increment in departure and arrival velocities."""
    # Get position and velocities for departure and arrival
    rr_dpt_body, vv_dpt_body =  departure_body.rv(epochs=t_launch)
    rr_arr_body, vv_arr_body =  target_body.rv(epochs=t_arrival)

    # Transform into Orbit objects
    orb_dpt = Orbit.from_vectors(
        Sun, rr_dpt_body, vv_dpt_body, epoch=t_launch
    )
    orb_arr = Orbit.from_vectors(
        Sun, rr_arr_body, vv_arr_body, epoch=t_arrival
    )

    # Define time of flight
    tof = orb_arr.epoch - orb_dpt.epoch

    if tof.to_value(u.s) <= 0:
        return None, None, None, None, None

    try:
        # Lambert is now a Maneuver object
        man_lambert = Maneuver.lambert(orb_dpt, orb_arr, prograde=prograde)

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


def targeting(departure_body, target_body, launch_span, arrival_span, prograde):
    args_list = [
        (departure_body, target_body, t_launch, t_arrival, prograde)
        for t_launch in launch_span for t_arrival in arrival_span
    ]
    with Pool() as pool:
        results = pool.starmap(_targetting, args_list)
    return results


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
    prograde: bool
        Prograde or retrograde transfer.
    """
    def __init__(
        self,
        departure_body,
        target_body,
        launch_span,
        arrival_span,
        prograde=True,
    ):
        self.departure_body = departure_body
        self.target_body = target_body
        self.launch_span = launch_span
        self.arrival_span = arrival_span
        self.prograde = prograde

        results = targeting(
            self.departure_body,
            self.target_body,
            self.launch_span,
            self.arrival_span,
            self.prograde,
        )

        dv_launch, dv_arrival, c3_launch, c3_arrival, tof = zip(*results)
        self.launch_dv = np.array(dv_launch).reshape(len(self.launch_span), len(self.arrival_span)).T
        self.arrival_dv = np.array(dv_arrival).reshape(len(self.launch_span), len(self.arrival_span)).T
        self.c3_launch = np.array(c3_launch).reshape(len(self.launch_span), len(self.arrival_span)).T
        self.c3_arrival = np.array(c3_arrival).reshape(len(self.launch_span), len(self.arrival_span)).T
        self.tof = np.array(tof).reshape(len(self.launch_span), len(self.arrival_span)).T

        # # Compute the minimum c3 energy value for launch and its associated
        # # launch and arrival dates
        # self.min_c3_launch = self.c3_launch.min()
        # self.launch_date_at_min_c3_launch = np.meshgrid(self.launch_span, self.arrival_span)[0][np.unravel_index(self.c3_launch.argmin(), self.c3_launch.shape)]
        # self.arrival_date_at_min_c3_launch = np.meshgrid(self.launch_span, self.arrival_span)[1][np.unravel_index(self.c3_launch.argmin(), self.c3_launch.shape)]

        # # Compute the minimum c3 energy value for arrival and its associated
        # # launch and arrival dates
        # self.min_c3_arrival = self.c3_arrival.min()
        # self.launch_date_at_min_c3_arrival = np.meshgrid(self.launch_span, self.arrival_span)[0][np.unravel_index(self.c3_arrival.argmin(), self.c3_arrival.shape)]
        # self.arrival_date_at_min_c3_arrival = np.meshgrid(self.launch_span, self.arrival_span)[1][np.unravel_index(self.c3_arrival.argmin(), self.c3_arrival.shape)]

    def _setup_plot(self, ax):
        """Setup the plot.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes object for plotting.
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(15, 15))
        else:
            self.ax = ax
            self.fig = self.ax.figure

        self.ax.set_xlabel("Launch date", fontsize=10, fontweight="bold")
        self.ax.set_ylabel("Arrival date", fontsize=10, fontweight="bold")

    def plot_launch_energy(
        self,
        levels=np.linspace(0, 45, 30) * u.km**2 / u.s**2,
        plot_contour_lines=True,
        ax=None,
    ):
        """Plot the characteristic launch energy.

        Parameters
        ----------
        levels: numpy.ndarray
            Levels for c3 contour lines.
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes object for plotting.

        """
        self._setup_plot(ax)

        c3_launch_colors = self.ax.contourf(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            self.c3_launch.astype("float64"),
            levels.astype("float64"),
        )
        c3_colorbar = self.fig.colorbar(c3_launch_colors)
        c3_colorbar.set_label("$km^2 / s^2$")

        # Draw the solid contour lines on top of previous colors
        if plot_contour_lines:
            c3_launch_lines = self.ax.contour(
                [D.to_datetime() for D in self.launch_span],
                [A.to_datetime() for A in self.arrival_span],
                self.c3_launch.astype("float64"),
                levels.astype("float64"),
                colors="black",
                linestyles="solid",
            )
            self.ax.clabel(c3_launch_lines, inline=1, fmt="%1.1f", colors="k", fontsize=10)

    def plot_arival_energy(
        self,
        levels=np.linspace(0, 45, 30) * u.km**2 / u.s**2,
        plot_contour_lines=True,
        ax=None,
    ):
        """Plot the characteristic energy at arrival.

        Parameters
        ----------
        levels: numpy.ndarray
            Levels for c3 contour lines.
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes object for plotting.

        """
        self._setup_plot(ax)

        c3_arrival_colors = self.ax.contourf(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            self.c3_launch.astype("float64"),
            levels.astype("float64"),
        )
        c3_colorbar = self.fig.colorbar(c3_arrival_colors)
        c3_colorbar.set_label("$km^2 / s^2$")

        if plot_contour_lines:
            c3_arrival_lines = self.ax.contour(
                [D.to_datetime() for D in self.launch_span],
                [A.to_datetime() for A in self.arrival_span],
                self.c3_launch.astype("float64"),
                levels.astype("float64"),
                colors="black",
                linestyles="solid",
            )
            self.ax.clabel(c3_arrival_lines, inline=1, fmt="%1.1f $km^2/s^2$", colors="k", fontsize=10)

    def plot_time_of_flight(self, levels, ax=None):
        """Plot the arrival velocity.

        Parameters
        ----------
        levels: numpy.ndarray
            Levels for time of flight contour lines.
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes object for plotting.

        """
        self._setup_plot(ax)

        tof_lines = self.ax.contour(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            (self.tof / 365.25).astype("float64"),
            levels.to_value(u.year).astype("float64"),
            colors="red",
            linestyles="dashed",
            linewidths=3.5,
        )
        tof_lines.set(path_effects=[patheffects.withStroke(linewidth=6, foreground="w")])
        tof_lines_labels = self.ax.clabel(
            tof_lines, inline=True, fmt="%1.0f years", colors="red", fontsize=14, use_clabeltext=True
        )
        plt.setp(tof_lines_labels, path_effects=[
            patheffects.withStroke(linewidth=3, foreground="w")])

    def plot_launch_velocity(self, launch_velocity_levels, ax=None):
        """Plot the arrival velocity.

        Parameters
        ----------
        launch_veocity_levels: numpy.ndarray
            Levels for launch velocity contour lines.
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes object for plotting.

        """
        self._setup_plot(ax)

        departure_dv_lines = self.ax.contour(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            self.launch_dv.astype("float64"),
            launch_velocity_levels.astype("float64"),
            colors="red",
            linewidths=3.5,
        )
        departure_dv_lines.set(path_effects=[patheffects.withStroke(linewidth=6, foreground="w")])
        departure_dv_labels = self.ax.clabel(
            departure_dv_lines, inline=True, fmt="%1.1f km/s", colors="red", fontsize=14, use_clabeltext=True
        )
        plt.setp(departure_dv_labels, path_effects=[
            patheffects.withStroke(linewidth=3, foreground="w")])

    def plot_arrival_velocity(self, levels, ax=None):
        """Plot the arrival velocity.

        Parameters
        ----------
        levels: numpy.ndarray
            Levels for arrival velocity contour lines.
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes object for plotting.

        """
        self._setup_plot(ax)

        arrival_dv_lines = self.ax.contour(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            self.arrival_dv.astype("float64"),
            levels.astype("float64"),
            colors="red",
            linewidths=3.5,
        )
        arrival_dv_lines.set(path_effects=[patheffects.withStroke(linewidth=6, foreground="w")])
        arrival_dv_labels = self.ax.clabel(
            arrival_dv_lines, inline=True, fmt="%1.1f km/s", colors="red", fontsize=14, use_clabeltext=True
        )
        plt.setp(arrival_dv_labels, path_effects=[
            patheffects.withStroke(linewidth=3, foreground="w")])

    def show(self):
        """Show the porkchop plot."""
        plt.show()
